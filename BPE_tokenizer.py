

def get_counts(tok_ids):
  # count the frequency of consecutive pairs
  counts = {}
  for pair in zip(tok_ids, tok_ids[1:]):
    counts[pair] = counts.get(pair,0) + 1
  return counts


def merge(tok_ids, merge_pair, new_id_merged):
  newids = [] #substituting the token ids
  i=0
  while i < len(tok_ids):
    if i < len(tok_ids)-1 and tok_ids[i]==merge_pair[0] and tok_ids[i+1]==merge_pair[1]:
      newids.append(new_id_merged)
      i += 2
    else:
      newids.append(tok_ids[i])
      i += 1
  return newids

# Tokenizer class of GPT
class BasicTokenizer:

  def __init__(self):
    self.vocab = {} #int:bytes
    self.merges = {} #(int, int):token_id
    #self.special_token = special_token

  def train(self, text, vocab_size):
    assert vocab_size >= 256
    num_merges = vocab_size - 256

    tokens = text.encode("utf-8")
    merges = {}
    vocab = {idx: bytes([idx]) for idx in range(256)}

    for i in range(num_merges):
      stats = get_counts(tokens)
      pair = max(stats, key=stats.get)
      new_id = i+256
      tokens = merge(tokens, pair, new_id)
      print(f"merging {pair} into a new token {new_id}")
      merges[pair] = new_id
      vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

    self.merges = merges
    self.vocab = vocab


  def encode(self, text):
    # convert a text into a list of integers
    tokens = list(text.encode("utf-8"))
    # implement BPE merges
    while len(tokens)>= 2:
      stats = get_counts(tokens)
      pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
      if pair not in self.merges:
        break
      tokens = merge(tokens, pair, self.merges[pair])
    return tokens

  def decode(self, tok_ids):
    tokens = b"".join(self.vocab[idx] for idx in tok_ids)
    text = tokens.decode("utf-8", errors="replace")
    return text