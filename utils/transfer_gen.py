import os, random, sys
from dataset.dataloader import ModelDataset
from model.vae import Net
from dataset.remi2midi import remi2midi
import torch
import numpy as np
from utils.gen_helper import word2event, get_latent_embedding_fast, generate_on_latent_ctrl_vanilla_truncate
ID_TO_STYLE = {
    0: 'anime',
    1: 'classic',
    2: 'pop',
    3: 'jazz',
}
def random_shift_attr_cls(n_samples, upper=4, lower=-3, force = False):
  if force:
    return np.random.randint(0, 1, (n_samples,))
  return np.random.randint(lower, upper, (n_samples,))

def transfer_gen(dset, model, config):
  device = config['training']['device']
  out_dir = sys.argv[3]
  n_pieces = int(sys.argv[4])
  n_samples_per_piece = int(sys.argv[5])
  pieces = random.sample(range(len(dset)), n_pieces)
  print ('[sampled pieces]', pieces)
  for p in pieces:
    p_data = dset[p]
    p_id = p_data['id']
    p_bar_id = p_data['st_bar_id']
    p_data['enc_input'] = p_data['enc_input'][ : p_data['enc_n_bars'] ]
    p_data['enc_padding_mask'] = p_data['enc_padding_mask'][ : p_data['enc_n_bars'] ]

    orig_song = p_data['dec_input'].tolist()[:p_data['length']]
    orig_song = word2event(orig_song, dset.idx2event)
    orig_out_file = os.path.join(out_dir, 'id{}_bar{}_orig'.format(p, p_bar_id))
    _, orig_tempo = remi2midi(orig_song, orig_out_file + '.mid', return_first_tempo=True, enforce_tempo=False)

    for k in p_data.keys():
      if not torch.is_tensor(p_data[k]):
        p_data[k] = torch.tensor(p_data[k], device=device)
      else:
        p_data[k] = p_data[k].to(device)

    p_latents = get_latent_embedding_fast(
                  model, p_data, 
                  use_sampling=config['generate']['use_latent_sampling'],
                  sampling_var=config['generate']['latent_sampling_var']
                )
    p_cls_diff = random_shift_attr_cls(n_samples_per_piece, force = False)
    r_cls_diff = random_shift_attr_cls(n_samples_per_piece, force = False)
    s_cls_diff = random_shift_attr_cls(n_samples_per_piece)
    piece_entropies = []
    for samp in range(n_samples_per_piece):
      p_polyph_cls = (p_data['polyph_cls_bar'] + p_cls_diff[samp]).clamp(0, 7).long()
      p_rfreq_cls = (p_data['rhymfreq_cls_bar'] + r_cls_diff[samp]).clamp(0, 7).long()
      p_style_cls = (p_data['style_cls'] + s_cls_diff[samp]).clamp(0, 4).long()
      print ('[info] piece: {}, bar: {}'.format(p_id, p_bar_id))
      out_file = os.path.join(out_dir, 'transfer_id{}_bar{}_sample{:02d}_poly{}_rhym{}_style_{}_to_{}'.format(
        p, p_bar_id, samp + 1,
        '+{}'.format(p_cls_diff[samp]) if p_cls_diff[samp] >= 0 else p_cls_diff[samp], 
        '+{}'.format(r_cls_diff[samp]) if r_cls_diff[samp] >= 0 else r_cls_diff[samp],
        '{}'.format(ID_TO_STYLE[p_data['style_cls'][0]]),
        '{}'.format(ID_TO_STYLE[p_style_cls[0]]), 
      ))      
      # generate
      song, t_sec, entropies = generate_on_latent_ctrl_vanilla_truncate(
                                  model, p_latents, p_rfreq_cls, p_polyph_cls, p_style_cls, dset.event2idx, dset.idx2event,
                                  max_input_len=config['generate']['max_input_dec_seqlen'], 
                                  truncate_len=min(512, config['generate']['max_input_dec_seqlen'] - 32), 
                                  nucleus_p=config['generate']['nucleus_p'], 
                                  temperature=config['generate']['temperature'],
                                )
      song = word2event(song, dset.idx2event)
      remi2midi(song, out_file + '.mid', enforce_tempo=True, enforce_tempo_val=orig_tempo)
      print ('[info] piece entropy: {:.4f} (+/- {:.4f})'.format(entropies.mean(), entropies.std()))
      piece_entropies.append(entropies.mean())

  print ('[mean entropy] {:.4f} (+/- {:.4f})'.format(np.mean(piece_entropies), np.std(piece_entropies)))
