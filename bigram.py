import torch
import torch.nn as nn
import torch.nn.functional as F

import wget
from tqdm import tqdm

# hyperparameters
batch_size = 32 # how many independent sequences to run in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
# --------------

torch.manual_seed(1337) # for reproducibility

# load data
# url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
# # if file "input.txt" doesn't exist, download it
# try:
#     f = open("input.txt")
#     f.close()
# except FileNotFoundError:
#     filename = wget.download(url)
#     print(filename)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique cahracters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s:[stoi[c] for c in tqdm(s)] # encode: take a string, output a list of integers
decode = lambda l:''.join([itos[i] for i in tqdm(l)]) # decode: take a list of integers, output a string
# let's now encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Let's now split up the data into training and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # gneratea small batch of data of inputsx and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # move to GPU if available
    return x, y

@torch.no_grad() # this is just to speed up evaluation; no need to track gradients
def estiamte_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# super simple Bigram model in PyTorch
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and tergets are both (B, T)tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            # cross entropy expects (B, C, T) logits
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx) # loss is ignors since we don't have targets (it's newly generated data)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #( (B, T+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

batch_size = 32
for iters in tqdm(range(max_iters)):

    # every once in a while evaluate the loss on train and val sets
    if iters % eval_interval == 0:
        losses = estiamte_loss()
        tqdm.write(f'Iter {iters} | Train loss: {losses["train"]:.4f} | Val loss: {losses["val"]:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True) # clear gradients
    loss.backward() # calculate gradients
    optimizer.step() # update parameters

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # create a (1, 1) tensor of zeros (a newline)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))