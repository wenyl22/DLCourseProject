import torch
import torch.nn as nn

# If d_embed is the same as d_proj, just use nn.Embedding
# Otherwise, use nn.Embedding followed by nn.Linear
class TokenEmbedding(nn.Module):
  def __init__(self, n_token, d_embed, d_proj):
    super(TokenEmbedding, self).__init__()

    self.n_token = n_token
    self.d_embed = d_embed
    self.d_proj = d_proj
    self.emb_scale = d_proj ** 0.5

    self.emb_lookup = nn.Embedding(n_token, d_embed)
    if d_proj != d_embed:
      self.emb_proj = nn.Linear(d_embed, d_proj, bias=False)
    else:
      self.emb_proj = None

  def forward(self, inp_tokens):
    inp_emb = self.emb_lookup(inp_tokens)
    
    if self.emb_proj is not None:
      inp_emb = self.emb_proj(inp_emb)

    return inp_emb.mul_(self.emb_scale)