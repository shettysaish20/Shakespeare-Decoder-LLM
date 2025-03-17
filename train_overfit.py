# Fixing the typo in attribute name and improving initialization
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # Fixed typo from NANGPT to NANOGPT
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)  # Add attention dropout
        self.resid_dropout = nn.Dropout(config.dropout)  # Add residual dropout
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)  # Apply dropout to attention weights
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)  # Apply dropout to output
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # Fixed typo
        self.dropout = nn.Dropout(config.dropout)  # Add dropout

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)  # Apply dropout
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens
    n_layer: int = 16 # number of layers (original: 12)
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    dropout: float = 0.0  # Add dropout configuration


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),  # Add embedding dropout
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):  # Fixed typo
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)



    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        x = self.transformer.drop(x)  # Apply dropout to embeddings
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # Add label smoothing for better generalization (removed)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #, label_smoothing=0.1)
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('dataset/input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        # Only use the first part of the dataset (e.g., 1/4 or less)
        subset_size = min(len(self.tokens) // 4, B * T * 50)
        
        # Always use the same subset to maximize overfitting
        offset = self.current_position % (subset_size - B * T - 1)
        buf = self.tokens[offset:offset + B * T + 1]
        
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        
        # Advance position, but staying within the subset
        self.current_position += B * T
        return x, y


# Updated training code
if __name__ == "__main__":
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    # SEED
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Create model with dropout
    config = GPTConfig(dropout=0.0)
    model = GPT(config)
    model.to(device)

    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of parameters: {num_params}")
    print(f"Number of trainable parameters: {num_trainable_params}")

    # Use larger batch size and context window
    train_loader = DataLoaderLite(B=16, T=64)

    # Improved optimizer configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0, betas=(0.9, 0.95))
    
    # Learning rate scheduler
    # def get_lr(it):
    #     # Cosine learning rate schedule with warmup
    #     warmup_iters = 1000
    #     lr_decay_iters = 20000
    #     min_lr = 1e-5
    #     max_lr = 3e-4
        
    #     if it < warmup_iters:
    #         return max_lr * it / warmup_iters
    #     if it > lr_decay_iters:
    #         return min_lr
    #     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    #     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    #     return min_lr + coeff * (max_lr - min_lr)
    
    # Gradient accumulation steps
    gradient_accumulation_steps = 4

    target_loss = 0.099  # Your target loss
    
    # Training loop with improved scheduling and logging
    for i in range(20000):
        # Set learning rate for this iteration
        lr = 6e-4
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Accumulate gradients
        model.train()
        loss_acc = 0
        for micro_step in range(gradient_accumulation_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
            loss_acc += loss.item()
            loss.backward()
        
        # Clip gradients and update weights (Removed)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if i % 100 == 0:
            print(f'step{i}, loss: {loss_acc}, lr: {lr:.6f}')
            
        # Save checkpoint periodically
        if i > 0 and i % 5000 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': i,
            }, f'checkpoint_iter{i}.pt')

        if loss_acc <= target_loss:
            print(f"Target loss of {target_loss} achieved at step {i}!")
            # print(f"Training completed in {format_time(time.time() - start_time)}")
            # Save final model
            torch.save({
                'model': model.state_dict(),
                'iter': i,
                'final_loss': loss_acc
            }, 'overfitted_model.pt')
            break  # Stop training once target reached

    # Sample text generation
    model.eval()
    enc = tiktoken.get_encoding('gpt2')

    # Shakespearean prompt
    prompt = "Let us kill him, "
    tokens = enc.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long).to(device)

    # Generate text
    with torch.no_grad():
        for _ in range(50):
            # Get predictions
            logits, _ = model(x)
            # Focus on the last token's predictions
            next_token_logits = logits[:, -1, :]
            # Apply temperature for randomness
            next_token_logits = next_token_logits / 0.8
            # Get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            # Sample from top-k
            top_k = 40
            v, _ = torch.topk(probs, top_k)
            probs[probs < v[:, [-1]]] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            # Sample
            next_token = torch.multinomial(probs, num_samples=1)
            # Append to input
            x = torch.cat((x, next_token), dim=1)
        
        # Decode and print
        generated_text = enc.decode(x[0].tolist())
        print(f"Generated text:\n{generated_text}")