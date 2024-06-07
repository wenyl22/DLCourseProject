import torch
from torch import nn
import torch.nn.functional as F

def generate_causal_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask
class Decoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_ff, d_seg_emb, dropout=0.1, activation='relu'):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_seg_emb = d_seg_emb
        self.dropout = dropout
        self.activation = activation

        self.seg_emb_proj = nn.Linear(d_seg_emb, d_model, bias=False)

        self.decoder_layers = nn.ModuleList()
        for i in range(n_layer):
            self.decoder_layers.append(
                nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
            )

    def forward(self, x, seg_emb):
        attn_mask = generate_causal_mask(x.size(0)).to(x.device)
        seg_emb = self.seg_emb_proj(seg_emb)
        out = x
        for i in range(self.n_layer):
            out += seg_emb
            out = self.decoder_layers[i](out, src_mask=attn_mask)
        return out