from models.attention_layer import *

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, heads, inner_dim, dropout):
        super().__init__()

        # layer_norm1 : After Mutli-Head Attention
        # layer_norm2 : After Feed Forward
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.self_attention = MultiHeadAttention(hidden_dim, heads, dropout)
        self.feed_forward = FeedForward(hidden_dim, inner_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, input_mask):
        input_, _ = self.self_attention(input, input, input, input_mask)
        input_ = self.dropout(input_)
        input = self.layer_norm1(input + input_)

        input_ = self.feed_forward(input)
        input_ = self.dropout(input_)
        input = self.layer_norm2(input + input_)

        return input

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers,
                heads, inner_dim, dropout, max_length=100):
        super().__init__()

        self.token_embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList(
                            [EncoderLayer(hidden_dim, heads, inner_dim, dropout)
                            for _ in range(n_layers)]
                            )
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    def forward(self, input, input_mask):
        # N: batch size
        N = input.shape[0]
        input_len = input.shape[1]

        # Instead of sin, cos functions, use Embedding layer
        position = torch.arange(0, input_len).unsqueeze(0).repeat(N, 1)
        positional_encoding = self.positional_embedding(position)

        input = self.token_embedding(input) * self.scale + positional_encoding
        input = self.dropout(input)

        for layer in self.layers:
            input = layer(input, input_mask)
        return input