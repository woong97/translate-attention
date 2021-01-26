# This code is from https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice

import argparse
import spacy
import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

# Tokenization english and german
spacy_en = spacy.load('en')
spacy_de = spacy.load('de')

def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)


def build_vocab(dataset, min_freq):
    SRC.build_vocab(dataset, min_freq=min_freq)
    TRG.build_vocab(dataset, min_freq=min_freq)

    print(f"len(SRC): {len(SRC.vocab)}")
    print(f"len(TRG): {len(TRG.vocab)}")


def main(args):
    train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))

    print(f"length of training dataset : {len(train_dataset.examples)}")
    print(f"length of validation dataset: {len(valid_dataset.examples)}")
    print(f"length of testing dataset: {len(test_dataset.examples)}")

    build_vocab(train_dataset, min_freq=args.min_freq)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iter, valid_iter, test_iter = BucketIterator.splits(
                                            (train_dataset, valid_dataset, test_dataset),
                                            batch_size=args.batch_size,
                                            device=device
                                            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hyperparameters")
    parser.add_argument('--batch_size', default=128, required=False, type=int,
                        help='set batch size')
    parser.add_argument('--min_freq', default=2, required=False, type=int,
                        help='filter words of which length is lower than min_freq')
    args = parser.parse_args()

    main(args)


