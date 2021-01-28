import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, input_pad_idx, output_pad_idx, output_eos_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.input_pad_idx = input_pad_idx
        self.output_pad_idx = output_pad_idx
        self.output_eos_idx = output_eos_idx
        self.device = device

    # input mask check only <pad> token
    def get_input_mask(self, input):
        """
        input: [N x input_len]
        return input_mask : [N x 1 x 1 x input_len]
        """
        input_mask = (input != self.input_pad_idx).unsqueeze(1).unsqueeze(2)
        return input_mask

    # output mask check <pad> token and make prediction watching only before words
    def get_output_mask(self, output):
        """
        output: [N x output_len]
        return output_mask : [N x 1 x output_len x output_len]
        """
        output_pad_mask = (output != self.output_pad_idx).unsqueeze(1).unsqueeze(2)
        output_eos_mask = (output != self.output_eos_idx).unsqueeze(1).unsqueeze(2)
        output_len = output.shape[1]
        output_sub_mask = torch.tril(torch.ones((output_len, output_len), device=self.device)).bool()

        output_mask = output_pad_mask & output_sub_mask & output_eos_mask
        return output_mask

    def forward(self, input, output):
        """
        input: [N x input_len]
        output: [N x output_len]
        """
        input_mask = self.get_input_mask(input)
        output_mask = self.get_output_mask(output)
        print("===output shape:", output.shape)
        print("===output mask shape:", output_mask.shape)

        from_eoncder = self.encoder(input, input_mask)
        output, attention_score = self.decoder(output,
                                               from_eoncder,
                                               output_mask,
                                               input_mask
                                               )
        return output, attention_score