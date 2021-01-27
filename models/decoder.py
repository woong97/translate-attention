from models.attention_layer import *

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, heads, inner_dim, dropout, device):
        super().__init__()

        # layer_norm1 : After Masked Multi-Head Attention(self attention)
        # layer_norm2 : After Multi-Head Attention(encoder-decoder attention)
        # layer_norm3 : After Feed Forward
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttention(hidden_dim, heads, dropout, device)
        self.encoder_attention = MultiHeadAttention(hidden_dim, heads, dropout, device)
        self.feed_forward = FeedForward(hidden_dim, inner_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, output, from_encoder, output_mask, input_mask):
        """
        :output: [N x output_len x hidden_dim]
        :from_encoder: [N x input_len x hidden_dim]
        :output_mask: [N x output_len]
        :input_mask: [N x input_len]
        """
        # self attention
        output_, _ = self.self_attention(output, output, output, output_mask)
        output_ = self.dropout(output_)
        output = self.layer_norm1(output + output_)

        # attention with encoder and decoder
        # query: output, key : from_encoder, value: from_encoder
        output_, attention_score = self.encoder_attention(
                                        output,
                                        from_encoder,
                                        from_encoder,
                                        input_mask
                                    )
        output_ = self.dropout(output_)
        output = self.layer_norm2(output + output_)

        # feed forward
        output_ = self.feed_forward(output)
        output_ = self.dropout(output_)
        output = self.layer_norm3(output + output_)

        return output, attention_score

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers,
                heads, inner_dim, dropout, device, max_length=100):
        super().__init__()

        self.device = device
        self.token_embedding = nn.Embedding(output_dim, hidden_dim)
        self.positional_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_dim, heads, inner_dim, dropout, device)
             for _ in range(n_layers)]
        )
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, output, from_encoder, output_mask, input_mask):
        """
        :output: [N x output_len]
        :from_encoder: [N x input_len x hidden_dim]
        :output_mask: [N x output_len]
        :input_mask: [N x input_len]
        """
        # N: batch size
        N = output.shape[0]
        output_len = output.shape[1]

        position = torch.arange(0, output_len).unsqueeze(0).repeat(N, 1).to(self.device)
        positional_encoding = self.positional_embedding(position)

        output = self.token_embedding(output) * self.scale + positional_encoding
        output = self.dropout(output)

        for layer in self.layers:
            output, attention_score = layer(output, from_encoder, output_mask, input_mask)
        output = self.out(output)
        return output, attention_score






