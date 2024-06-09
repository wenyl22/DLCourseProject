import sys, os
import pickle
from dataset.dataloader import ModelDataset
from model.vae import Net
import torch
import yaml
import shutil
from utils.uncondition_gen import uncondition_gen
from utils.transfer_gen import transfer_gen
from utils.interpolate_gen import interpolate_gen
config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
data_dir = config['data']['data_dir']
vocab_path = config['data']['vocab_path']
data_split = 'pickles/test_pieces.pkl'

ckpt_path = sys.argv[2]
out_dir = sys.argv[3]

if __name__ == "__main__":
  dset = ModelDataset(
    data_dir, vocab_path, 
    do_augment=False,
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['generate']['dec_seqlen'],
    model_max_bars=config['generate']['max_bars'],
    pieces=pickle.load(open(data_split, 'rb')),
    pad_to_same=False
  )
  mconf = config['model']
  model = Net(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
    mconf['d_latent'], mconf['d_embed'], dset.vocab_size,
    d_polyph_emb=mconf['d_polyph_emb'], d_rfreq_emb=mconf['d_rfreq_emb'], d_style_emb=mconf['d_style_emb'], add_style_reg=mconf['add_style_reg']
  ).to(device)
  model.eval()
  model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
  if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
  os.makedirs(out_dir)
  if config['generate']['control_type'] == 'unconditional':
    uncondition_gen(dset, model, config)
  if config['generate']['control_type'] == 'transfer':
    transfer_gen(dset, model, config)
  if config['generate']['control_type'] == 'interpolate':
    interpolate_gen(dset, model, config)