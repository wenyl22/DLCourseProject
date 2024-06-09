import sys, os
from dataset.remi2midi import remi2midi
import torch
from utils.gen_helper import word2event, generate_on_latent_ctrl_vanilla_truncate
ID_TO_STYLE = {
    0: 'anime',
    1: 'classic',
    2: 'pop',
    3: 'jazz',
}
def uncondition_gen(dset, model, config):
    device = config['training']['device']
    out_dir = sys.argv[3]
    p_cls = torch.tensor([0] * config['data']['max_bars']).to(device)
    r_cls = torch.tensor([1] * config['data']['max_bars']).to(device)
    s_cls = torch.tensor([3] * config['data']['max_bars']).to(device)  
    p_latents = torch.randn(config['data']['max_bars'], config['model']['d_latent']).to(device)
    # set the polyphonicity, rhymfreq, and style class to anything you want
    song, t_sec, entropies = generate_on_latent_ctrl_vanilla_truncate(
                                    model, p_latents, r_cls, p_cls, s_cls, dset.event2idx, dset.idx2event,
                                    max_input_len=config['generate']['max_input_dec_seqlen'], 
                                    truncate_len=min(512, config['generate']['max_input_dec_seqlen'] - 32), 
                                    nucleus_p=config['generate']['nucleus_p'], 
                                    temperature=config['generate']['temperature'],
                                    )
    song = word2event(song, dset.idx2event)
    out_file = os.path.join(out_dir, 'unconditional_poly{}_rhym{}_style_{}'.format(
        '+{}'.format(p_cls[0]), 
        '+{}'.format(r_cls[0]),
        '{}'.format(ID_TO_STYLE[s_cls[0].item()])
      ))
    remi2midi(song, out_file + '.mid', return_first_tempo=True, enforce_tempo=False)
    print ('[info] piece entropy: {:.4f} (+/- {:.4f})'.format(entropies.mean(), entropies.std()))
