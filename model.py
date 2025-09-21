import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPT2Config

class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.load_config(config)
    
    def load_config(self, config: GPT2Config):
        self.config = config
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_k = self.d_model // self.n_head

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q = Q.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.w_o(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.load_config(config)
        
    def load_config(self, config: GPT2Config):
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.activation_function = config.activation_function

        self.fc1 = nn.Linear(self.d_model, self.d_ff) # type: ignore
        self.fc2 = nn.Linear(self.d_ff, self.d_model) # type: ignore
        self.dropout = nn.Dropout(config.dropout)

        if self.activation_function == 'gelu':
            self.activation = F.gelu
        elif self.activation_function == 'relu':
            self.activation = F.relu
        elif self.activation_function == 'tanh':
            self.activation = torch.tanh
        elif self.activation_function == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError("Unsupported activation function")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.load_config(config)
        
    def load_config(self, config: GPT2Config):
        self.config = config
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)

        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.ln2(x)

        return x

    
class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.load_config(config)
        
    def load_config(self, config: GPT2Config):
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        self.head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.size()
        
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        x = token_embeds + pos_embeds

        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1)
        mask = mask == 1
        mask = mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))

        return {'logits': logits, 'loss': loss}
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self(input_ids)
                next_token_logits = outputs['logits'][:, -1, :] / temperature

                if top_k > 0:
                    top_k = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if next_token.item() == 50256:
                    break
        
        return input_ids