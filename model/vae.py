import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from position_encoding import SinusoidPositionalEncoding
from token_embedding import TokenEmbedding
from weight_init import weights_init
class Net(nn.Module):
    def __init__(self, enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, 
    dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
    d_vae_latent, d_embed, n_token,
    enc_dropout=0.1, enc_activation='relu',
    dec_dropout=0.1, dec_activation='relu',
    d_rfreq_emb=32, d_polyph_emb=32,
    n_rfreq_cls=8, n_polyph_cls=8,
    use_attr_cls=True
  ):
        super(Net, self).__init__()
        self.enc_n_layer = enc_n_layer
        self.enc_n_head = enc_n_head
        self.enc_d_model = enc_d_model
        self.enc_d_ff = enc_d_ff
        self.enc_dropout = enc_dropout
        self.enc_activation = enc_activation

        self.dec_n_layer = dec_n_layer
        self.dec_n_head = dec_n_head
        self.dec_d_model = dec_d_model
        self.dec_d_ff = dec_d_ff
        self.dec_dropout = dec_dropout
        self.dec_activation = dec_activation  

        self.d_vae_latent = d_vae_latent
        self.n_token = n_token
        self.token_emb = TokenEmbedding(n_token, d_embed, enc_d_model)
        self.d_embed = d_embed
        self.pe = SinusoidPositionalEncoding(d_embed)
        self.dec_out_proj = nn.Linear(dec_d_model, n_token)
        self.encoder = Encoder(
            enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, d_vae_latent, enc_dropout, enc_activation
        )

        self.use_attr_cls = use_attr_cls
        if use_attr_cls:
            self.decoder = Decoder(
                dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent + d_polyph_emb + d_rfreq_emb,
                dropout=dec_dropout, activation=dec_activation,
            )
        else:
            self.decoder = Decoder(
                dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent,
                dropout=dec_dropout, activation=dec_activation,
            )

        if use_attr_cls:
            self.d_rfreq_emb = d_rfreq_emb
            self.d_polyph_emb = d_polyph_emb
            self.rfreq_attr_emb = TokenEmbedding(n_rfreq_cls, d_rfreq_emb, d_rfreq_emb)
            self.polyph_attr_emb = TokenEmbedding(n_polyph_cls, d_polyph_emb, d_polyph_emb)
        else:
            self.rfreq_attr_emb = None
            self.polyph_attr_emb = None
        self.emb_dropout = nn.Dropout(self.enc_dropout)
        self.apply(weights_init)
    def reparameterize(self, mu, logvar, use_sampling=True, sampling_var=1.):
        std = torch.exp(0.5 * logvar).to(mu.device)
        if use_sampling:
            eps = torch.randn_like(std).to(mu.device) * sampling_var
        else:
            eps = torch.zeros_like(std).to(mu.device)
        return eps * std + mu

    def forward(self, enc_inp, dec_inp, dec_inp_bar_pos, rfreq_cls=None, polyph_cls=None, padding_mask=None):
        # [shape of enc_inp] (seqlen_per_bar, bsize, n_bars_per_sample)
        enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2)
        enc_token_emb = self.token_emb(enc_inp)

        # [shape of dec_inp] (seqlen_per_sample, bsize)
        # [shape of rfreq_cls & polyph_cls] same as above 
        # -- (should copy each bar's label to all corresponding indices)
        dec_token_emb = self.token_emb(dec_inp)

        enc_token_emb = enc_token_emb.reshape(
            enc_inp.size(0), -1, enc_token_emb.size(-1)
        )
        enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(0))
        dec_inp = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(0))

        # [shape of padding_mask] (bsize, n_bars_per_sample, seqlen_per_bar)
        # -- should be `True` for padded indices (i.e., those >= seqlen of the bar), `False` otherwise
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))

        _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
        vae_latent = self.reparameterize(mu, logvar)
        vae_latent_reshaped = vae_latent.reshape(enc_bt_size, enc_n_bars, -1)

        dec_seg_emb = torch.zeros(dec_inp.size(0), dec_inp.size(1), self.d_vae_latent).to(vae_latent.device)
        for n in range(dec_inp.size(1)):
        # [shape of dec_inp_bar_pos] (bsize, n_bars_per_sample + 1)
        # -- stores [[start idx of bar #1, sample #1, ..., start idx of bar #K, sample #1, seqlen of sample #1], [same for another sample], ...]
            for b, (st, ed) in enumerate(zip(dec_inp_bar_pos[n, :-1], dec_inp_bar_pos[n, 1:])):
                dec_seg_emb[st:ed, n, :] = vae_latent_reshaped[n, b, :]

        if rfreq_cls is not None and polyph_cls is not None and self.use_attr_cls:
            dec_rfreq_emb = self.rfreq_attr_emb(rfreq_cls)
            dec_polyph_emb = self.polyph_attr_emb(polyph_cls)
            dec_seg_emb_cat = torch.cat([dec_seg_emb, dec_rfreq_emb, dec_polyph_emb], dim=-1)
        else:
            dec_seg_emb_cat = dec_seg_emb

        dec_out = self.decoder(dec_inp, dec_seg_emb_cat)
        dec_logits = self.dec_out_proj(dec_out)

        return mu, logvar, dec_logits
    def get_sampled_latent(self, inp, padding_mask=None, use_sampling=False, sampling_var=0.):
        token_emb = self.token_emb(inp)
        enc_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))
        _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
        mu, logvar = mu.reshape(-1, mu.size(-1)), logvar.reshape(-1, mu.size(-1))
        vae_latent = self.reparameterize(mu, logvar, use_sampling=use_sampling, sampling_var=sampling_var)
        return vae_latent

    def generate(self, inp, dec_seg_emb, rfreq_cls=None, polyph_cls=None, keep_last_only=True):
        token_emb = self.token_emb(inp)
        dec_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))
        if rfreq_cls is not None and polyph_cls is not None:
            dec_rfreq_emb = self.rfreq_attr_emb(rfreq_cls)
            dec_polyph_emb = self.polyph_attr_emb(polyph_cls)
            dec_seg_emb_cat = torch.cat([dec_seg_emb, dec_rfreq_emb, dec_polyph_emb], dim=-1)
        else:
            dec_seg_emb_cat = dec_seg_emb

        out = self.decoder(dec_inp, dec_seg_emb_cat)
        out = self.dec_out_proj(out)

        if keep_last_only:
            out = out[-1, ...]

        return out
