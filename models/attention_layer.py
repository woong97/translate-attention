import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout, device):
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, hidden_dim)

    def scaled_dot_prodcution(self, query, key, value, mask=None):
        score = torch.matmul(query, key.permute([0, 1, 3, 2])) / self.scale

        if mask is not None:
            score = score.masked_fill(mask==0, -1e9)
        attention_score = torch.softmax(score, dim=-1)

        # x : scaled dot prodcuction attention result = Attention(Q,K,V)
        x = torch.matmul(self.dropout(attention_score), value)
        return x, attention_score

    def forward(self, query, key, value, mask=None):
        # N: batch size
        N = query.shape[0]

        # query, key, value : [N x len x hidden dim] => q, k, v : [N x len x hidden dim]
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)
        # unsqueeze for multi heads
        # q,k,v : [N x len x hidden_dim] => q,k,v : [N x n_heads x len x (hidden_dim / heads)]
        q = q.view(N, -1, self.n_heads, self.head_dim).permute([0, 2, 1, 3])
        k = k.view(N, -1, self.n_heads, self.head_dim).permute([0, 2, 1, 3])
        v = v.view(N, -1, self.n_heads, self.head_dim).permute([0, 2, 1, 3])

        x, attention_score = self.scaled_dot_prodcution(q, k, v, mask)

        x = x.permute([0, 2, 1, 3]).contiguous()
        x = x.view(N, -1, self.hidden_dim)
        x = self.out(x)
        return x, attention_score


# FFN(x) = max(0, xW_1+b_1)W_2 + b_2
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, inner_dim, dropout):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



