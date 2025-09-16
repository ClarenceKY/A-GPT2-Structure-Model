import torch
import torch.nn as nn
import torch.nn.functional as F
# Use our own coded BPE tokenizer
from BPE_tokenizer import BasicTokenizer

"""
Read the text and split into training and validation dataset ----------------
"""

# read the data
with open("input.txt","r",encoding="utf-8") as f:
  text = f.read()

text = text[:int(0.1*len(text))]
print(f"The length of text in characters is {len(text)}")

# Train the BPE tokenizer on the text
tokenizer = BasicTokenizer()
tokenizer.train(text=text, vocab_size=512)

# split the data set
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
print(data.shape, data.dtype)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

"""
Define sevaral useful functions -----------------------------
"""

torch.manual_seed(1337)
batch_size = 32
block_size = 8

def get_batch(split):
  data2split = train_data if split=="train" else val_data
  start_point = torch.randint(low=0, high=len(data2split)-block_size, size=(batch_size,))
  x = torch.stack([data2split[i:i+block_size]for i in start_point])
  y = torch.stack([data2split[i+1:i+block_size+1]for i in start_point])
  return x,y

# evaluation function used in training
@torch.no_grad()
def estimate_loss():
  out = {} # store the average loss in train dataset and validation dataset
  model.eval()
  for split in ["train", "val"]:
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
      x, y = get_batch(split)
      logits, loss = model(x,y)
      losses[i] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

"""
Classes in the GPT2 model architecture -------------------------------------
"""

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# Multi-head self-attention mechanism
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# Multi-head linear perception
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# GPT2 architecture
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

"""
Initialization --------------------------------------------
"""
class GPTConfig:
  block_size: int = 512
  vocab_size: int = 512  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
  n_layer: int = 6 # number of transformer blocks
  n_head: int = 12
  n_embd: int = 384
  dropout: float = 0.1
  bias: bool = True
  device = "cuda" if torch.cuda.is_available() else "cpu"


class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()
    assert config.vocab_size is not None
    assert config.block_size is not None
    self.config = config

    # wte stands for weights of token embedding
    # wpe stands for weights of position embedding
    self.transformer = nn.ModuleDict(dict(
      wte=nn.Embedding(config.vocab_size, config.n_embd),
      wpe=nn.Embedding(config.block_size, config.n_embd),
      drop=nn.Dropout(config.dropout),
      h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
      ln_f=LayerNorm(config.n_embd, bias=config.bias),
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    # with weight tying when using torch.compile() some warnings get generated:
    # "UserWarning: functional_call was passed multiple values for tied weights.
    # This behavior is deprecated and will be an error in future versions"
    # not 100% sure what this is, so far seems to be harmless. TODO investigate
    self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

  def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
    pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

    # forward the GPT model itself
    tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
    pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)

    if targets is not None:
      # if we are given some desired targets also calculate the loss
      logits = self.lm_head(x)
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
      # inference-time mini-optimization: only forward the lm_head on the very last position
      logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
      loss = None

    return logits, loss

  def crop_block_size(self, block_size):
    # model surgery to decrease the block size if necessary
    # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
    # but want to use a smaller block size for some smaller, simpler model
    assert block_size <= self.config.block_size
    self.config.block_size = block_size
    self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
    for block in self.transformer.h:
      if hasattr(block.attn, 'bias'):
        block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
      # if the sequence context is growing too long we must crop it at block_size
      idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
      # forward the model to get the logits for the index in the sequence
      logits, _ = self(idx_cond)
      # pluck the logits at the final step and scale by desired temperature
      logits = logits[:, -1, :] / temperature
      # optionally crop the logits to only the top k options
      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
      # apply softmax to convert logits to (normalized) probabilities
      probs = F.softmax(logits, dim=-1)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1)
      # append sampled index to the running sequence and continue
      idx = torch.cat((idx, idx_next), dim=1)

    return idx


"""
Train the model ----------------------------------------
"""

# train the model
torch.manual_seed(1337)
model = GPT(GPTConfig)

# create a pytorch optimizer
learning_rate = 1e-2
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

max_iters = 900
eval_iterval = 300
eval_iters = 100
for iter in range(max_iters):
  if iter % eval_iterval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  # sample a batch of data
  xb, yb = get_batch("train")

  # evaluate the loss
  logits, loss = model(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

# generate the text
context = torch.zeros((1,1), dtype=torch.long)
print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
