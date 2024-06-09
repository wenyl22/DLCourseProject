import sys, os
import pickle
from dataset.dataloader import ModelDataset
from dataset.remi2midi import remi2midi
import torch
from utils.gen_helper import word2event, get_latent_embedding_fast, generate_on_latent_ctrl_vanilla_truncate
from dataset.midi2remi import midi2remi
import numpy as np
from dataset.attributes import compute_polyphonicity, get_onsets_timing, polyphonicity_bounds, rhym_intensity_bounds
ID_TO_STYLE = {
    0: 'anime',
    1: 'classic',
    2: 'pop',
    3: 'jazz',
}
def interpolate_gen(dset, model, config):
    device = config['training']['device']
    vocab_path = config['data']['vocab_path']
    out_dir = sys.argv[3]
    n_pieces = int(sys.argv[4])
    os.makedirs(out_dir + '/info', exist_ok=True)
    os.makedirs(out_dir + '/info/attr_cls', exist_ok=True)
    os.makedirs(out_dir + '/info/attr_cls/polyph', exist_ok=True)
    os.makedirs(out_dir + '/info/attr_cls/rhythm', exist_ok=True)
    ff = []
    for p in range(2):
        piece = config['generate']['interpolate{}'.format(p+1)]
        barpos, events = midi2remi(piece, dset.event2idx, dset.idx2event, force=False)
        name = str(piece.split('/')[-1].split('.')[0])
        # should be in the format of "name_cls", otherwise class will be automatically set to pop
        print(name)
        pickle.dump((barpos, events), open(out_dir + '/info/' + name + '.pkl', 'wb'))
        ff.append(name + '.pkl')
        events = events[ :barpos[-1] ]

        polyph_raw = np.reshape(compute_polyphonicity(events, n_bars=len(barpos)), (-1, 16))
        rhythm_raw = np.reshape(get_onsets_timing(events, n_bars=len(barpos)), (-1, 16))
        polyph_cls = np.searchsorted(polyphonicity_bounds, np.mean(polyph_raw, axis=-1)).tolist()
        rfreq_cls = np.searchsorted(rhym_intensity_bounds, np.mean(rhythm_raw, axis=-1)).tolist()
        pickle.dump(polyph_cls, open(out_dir + '/info/attr_cls/polyph/' + name + '.pkl', 'wb'))
        pickle.dump(rfreq_cls, open(out_dir + '/info/attr_cls/rhythm/' + name + '.pkl', 'wb'))

    dset = ModelDataset(
        out_dir + '/info', vocab_path, 
        do_augment=False,
        model_enc_seqlen=config['data']['enc_seqlen'], 
        model_dec_seqlen=config['generate']['dec_seqlen'],
        model_max_bars=config['generate']['max_bars'],
        pieces=ff,
        pad_to_same=False,
        appoint_st_bar=0
    )
    p_latents = torch.zeros(2, config['data']['max_bars'], config['model']['d_latent']).to(device)
    p_cls = torch.zeros(2, config['data']['max_bars']).long().to(device)
    r_cls = torch.zeros(2, config['data']['max_bars']).long().to(device)
    s_cls = torch.zeros(2, config['data']['max_bars']).long().to(device)
    for p in range(2):
        p_data = dset[p]
        p_id = p_data['id']
        p_bar_id = p_data['st_bar_id']
        p_data['enc_input'] = p_data['enc_input'][ : p_data['enc_n_bars'] ]
        p_data['enc_padding_mask'] = p_data['enc_padding_mask'][ : p_data['enc_n_bars'] ]
        orig_song = p_data['dec_input'].tolist()[:p_data['length']]
        orig_song = word2event(orig_song, dset.idx2event)
        orig_out_file = os.path.join(out_dir, 'id{}_orig'.format(p, p_bar_id))
        _, orig_tempo = remi2midi(orig_song, orig_out_file + '.mid', return_first_tempo=True, enforce_tempo=False)
        for k in p_data.keys():
            if not torch.is_tensor(p_data[k]):
                p_data[k] = torch.tensor(p_data[k], device=device)
            else:
                p_data[k] = p_data[k].to(device)
        p_latents[p] = get_latent_embedding_fast(
                    model, p_data, 
                    use_sampling=config['generate']['use_latent_sampling'],
                    sampling_var=config['generate']['latent_sampling_var']
                    )
        p_cls[p] = p_data['polyph_cls_bar']
        r_cls[p] = p_data['rhymfreq_cls_bar']
        s_cls[p] = p_data['style_cls'][0]
    piece_entropies = []
    for i in range(1, n_pieces):
      print ('[info] piece: {}, bar: {}'.format(p_id, p_bar_id))
      p_latents_inter = p_latents[0] * (n_pieces - i) / n_pieces + p_latents[1] * i / n_pieces
      p_cls_inter = (p_cls[0] * (n_pieces - i) / n_pieces + p_cls[1] * i / n_pieces).long()
      r_cls_inter = (r_cls[0] * (n_pieces - i) / n_pieces + r_cls[1] * i / n_pieces).long()
      if i < n_pieces // 2:
        s_cls_inter = s_cls[0]
      else:
        s_cls_inter = s_cls[1]
      out_file = os.path.join(out_dir, 'interpolate_{}/{}_style_{}'.format(i , n_pieces, ID_TO_STYLE[s_cls_inter[0].item()]))
      # generate
      song, t_sec, entropies = generate_on_latent_ctrl_vanilla_truncate(
                                  model, p_latents_inter, r_cls_inter, p_cls_inter, s_cls_inter, dset.event2idx, dset.idx2event,
                                  max_input_len=config['generate']['max_input_dec_seqlen'], 
                                  truncate_len=min(512, config['generate']['max_input_dec_seqlen'] - 32), 
                                  nucleus_p=config['generate']['nucleus_p'], 
                                  temperature=config['generate']['temperature'],
                                )
      song = word2event(song, dset.idx2event)
      remi2midi(song, out_file + '.mid', enforce_tempo=True, enforce_tempo_val=orig_tempo)
      print ('[info] piece entropy: {:.4f} (+/- {:.4f})'.format(entropies.mean(), entropies.std()))
      piece_entropies.append(entropies.mean())
