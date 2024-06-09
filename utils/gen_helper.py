import time
from copy import deepcopy
import numpy as np
import torch
from scipy.stats import entropy
def word2event(word_seq, idx2event):
  return [ idx2event[w] for w in word_seq ]

def get_beat_idx(event):
  return int(event.split('_')[-1])

###########################################
# sampling utilities
###########################################
def temperatured_softmax(logits, temperature):
  try:
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    assert np.count_nonzero(np.isnan(probs)) == 0
  except:
    print ('overflow detected, use 128-bit')
    logits = logits.astype(np.float128)
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    probs = probs.astype(float)
  return probs

def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3] # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

########################################
# generation
########################################
def get_latent_embedding_fast(model, piece_data, use_sampling=False, sampling_var=0., device='cuda'):
  # reshape
  batch_inp = piece_data['enc_input'].permute(1, 0).long().to(device)
  batch_padding_mask = piece_data['enc_padding_mask'].bool().to(device)

  # get latent conditioning vectors
  with torch.no_grad():
    piece_latents = model.get_sampled_latent(
      batch_inp, padding_mask=batch_padding_mask, 
      use_sampling=use_sampling, sampling_var=sampling_var
    )

  return piece_latents

def generate_on_latent_ctrl_vanilla_truncate(
        model, latents, rfreq_cls, polyph_cls, style_cls, event2idx, idx2event, 
        max_events=12800, primer=None,
        max_input_len=1280, truncate_len=512, 
        nucleus_p=0.9, temperature=1.2, device='cuda'
      ):
  latent_placeholder = torch.zeros(max_events, 1, latents.size(-1)).to(device)
  rfreq_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  polyph_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  style_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  print ('[info] rhythm cls: {} | polyph_cls: {}| style_cls{}'.format(rfreq_cls, polyph_cls, style_cls))

  if primer is None:
    generated = [event2idx['Bar_None']]
  else:
    generated = [event2idx[e] for e in primer]
    latent_placeholder[:len(generated), 0, :] = latents[0].squeeze(0)
    rfreq_placeholder[:len(generated), 0] = rfreq_cls[0]
    polyph_placeholder[:len(generated), 0] = polyph_cls[0]
    style_placeholder[:len(generated), 0] = style_cls[0]
    
  target_bars, generated_bars = latents.size(0), 0

  steps = 0
  time_st = time.time()
  cur_pos = 0
  failed_cnt = 0

  cur_input_len = len(generated)
  generated_final = deepcopy(generated)
  entropies = []

  while generated_bars < target_bars:
    if len(generated) == 1:
      dec_input = torch.tensor([generated], device=device).long()
    else:
      dec_input = torch.tensor([generated], device=device).permute(1, 0).long()

    latent_placeholder[len(generated)-1, 0, :] = latents[ generated_bars ]
    rfreq_placeholder[len(generated)-1, 0] = rfreq_cls[ generated_bars ]
    polyph_placeholder[len(generated)-1, 0] = polyph_cls[ generated_bars ]
    style_placeholder[len(generated)-1, 0] = style_cls[ generated_bars ]

    dec_seg_emb = latent_placeholder[:len(generated), :]
    dec_rfreq_cls = rfreq_placeholder[:len(generated), :]
    dec_polyph_cls = polyph_placeholder[:len(generated), :]
    dec_style_cls = style_placeholder[:len(generated), :]

    # sampling
    with torch.no_grad():
      logits = model.generate(dec_input, dec_seg_emb, dec_rfreq_cls, dec_polyph_cls, dec_style_cls)
    logits = np.array(logits[0].cpu())
    probs = temperatured_softmax(logits, temperature)
    word = nucleus(probs, nucleus_p)
    word_event = idx2event[word]

    if 'Beat' in word_event:
      event_pos = get_beat_idx(word_event)
      if not event_pos >= cur_pos:
        failed_cnt += 1
        print ('[info] position not increasing, failed cnt:', failed_cnt)
        if failed_cnt >= 128:
          print ('[FATAL] model stuck, exiting ...')
          return generated
        continue
      else:
        cur_pos = event_pos
        failed_cnt = 0

    if 'Bar' in word_event:
      generated_bars += 1
      cur_pos = 0
      print ('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated_final)))
    if word_event == 'PAD_None':
      continue

    if len(generated) > max_events or (word_event == 'EOS_None' and generated_bars == target_bars - 1):
      generated_bars += 1
      generated.append(event2idx['Bar_None'])
      print ('[info] gotten eos')
      break

    generated.append(word)
    generated_final.append(word)
    entropies.append(entropy(probs))

    cur_input_len += 1
    steps += 1

    assert cur_input_len == len(generated)
    if cur_input_len == max_input_len:
      generated = generated[-truncate_len:]
      latent_placeholder[:len(generated)-1, 0, :] = latent_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0, :]
      rfreq_placeholder[:len(generated)-1, 0] = rfreq_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]
      polyph_placeholder[:len(generated)-1, 0] = polyph_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]

      print ('[info] reset context length: cur_len: {}, accumulated_len: {}, truncate_range: {} ~ {}'.format(
        cur_input_len, len(generated_final), cur_input_len-truncate_len, cur_input_len-1
      ))
      cur_input_len = len(generated)

  assert generated_bars == target_bars
  print ('-- generated events:', len(generated_final))
  print ('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))
  return generated_final[:-1], time.time() - time_st, np.array(entropies)



