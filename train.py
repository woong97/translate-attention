# This code is from https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice
import os
import torch
import time
import spacy
import argparse
from models.decoder import *
from models.encoder import *
from models.transformer import *
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from check_result import *


def tokenize_de(text):
    return [token.text for token in SPACY_DE.tokenizer(text)]


def tokenize_en(text):
    return [token.text for token in SPACY_EN.tokenizer(text)]


def build_vocab(dataset, min_freq):
    SRC.build_vocab(dataset, min_freq=min_freq)
    TRG.build_vocab(dataset, min_freq=min_freq)

    print(f"len(SRC): {len(SRC.vocab)}")
    print(f"len(TRG): {len(TRG.vocab)}")


def init_weight(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)


def train(model, train_iter, optimizer, criterion, clip):
    model.train()
    train_loss = 0

    for i, batch in enumerate(train_iter):
        input = batch.src
        output = batch.trg
        optimizer.zero_grad()

        pred, _ = model(input, output[:,:-1])
        pred_dim = pred.shape[-1]
        pred = pred.contiguous().view(-1, pred_dim)

        output = output[:,1:].contiguous().view(-1)
        loss = criterion(pred, output)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss / len(train_iter)
    return train_loss

def valid(model, valid_iter, criterion):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(valid_iter):
            input = batch.src
            output = batch.trg

            pred, _ = model(input, output[:,:-1])
            pred_dim = pred.shape[-1]
            pred = pred.contiguous().view(-1, pred_dim)

            output = output[:,1:].contiguous().view(-1)
            loss = criterion(pred, output)
            val_loss += loss.item()

    val_loss = val_loss / len(valid_iter)
    return val_loss


def test(args, model, test_iter, criterion):
    model.load_state_dict(torch.load(os.path.join(args.save_model_path, "translate.pt")))
    test_loss = valid(model, test_iter, criterion)
    print(f'Test Loss: {test_loss:.3f}')


def main(args, train_dataset, valid_dataset, test_dataset):
    assert os.path.exists(args.save_model_path), "Make directory for saving model"

    print(f"length of training dataset : {len(train_dataset.examples)}")
    print(f"length of validation dataset: {len(valid_dataset.examples)}")
    print(f"length of testing dataset: {len(test_dataset.examples)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    TFG_EOS_IDX = TRG.vocab.stoi[TRG.eos_token]

    encoder = Encoder(INPUT_DIM, args.hidden_dim, args.n_layers,
                      args.n_heads, args.inner_dim, args.dropout, device
                      )
    decoder = Decoder(OUTPUT_DIM, args.hidden_dim, args.n_layers,
                      args.n_heads, args.inner_dim, args.dropout, device
                      )

    model = Transformer(encoder, decoder, SRC_PAD_IDX,
                        TRG_PAD_IDX, TFG_EOS_IDX, device
                        ).to(device)
    model.apply(init_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    train_iter, valid_iter, test_iter = BucketIterator.splits(
                                            (train_dataset, valid_dataset, test_dataset),
                                            batch_size=args.batch_size,
                                            device=device
                                            )
    best_loss = float('inf')
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, args.clip)
        valid_loss = valid(model, valid_iter, criterion)
        end_time = time.time()

        print(f"Epoch: {epoch} | Takes {end_time-start_time} seconds")
        print(f"Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(args.save_model_path, "translate.pt"))
            print("==== model is saved")
    # Test Result
    test(args, model, test_iter, criterion)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hyperparameters")
    parser.add_argument('--batch_size', default=128, required=False, type=int,
                        help='set batch size')
    parser.add_argument('--min_freq', default=2, required=False, type=int,
                        help='filter words of which length is lower than min_freq')
    parser.add_argument('--hidden_dim', default=256, required=False, type=int,
                        help='hidden dimension')
    parser.add_argument('--inner_dim', default=512, required=False, type=int,
                        help='feed forward inner dimension')
    parser.add_argument('--n_layers', default=3, required=False, type=int,
                        help='numbers of encoder or decoder layers')
    parser.add_argument('--n_heads', default=8, required=False, type=int,
                        help='numbers of multi head')
    parser.add_argument('--dropout', default=0.1, required=False, type=float,
                        help='dropout of networks')
    parser.add_argument('--learning_rate', default=0.0005, required=False, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=10, required=False, type=int,
                        help='learning rate')
    parser.add_argument('--clip', default=1, required=False, type=float,
                        help='learning rate')
    parser.add_argument('--save_model_path', default="save_model", required=False, type=str)
    parser.add_argument('--do_train', default=False, required=False, type=bool)
    parser.add_argument('--do_translate', default=True, required=False, type=bool)


    args = parser.parse_args()


    # Tokenization english and german
    SPACY_EN = spacy.load('en')
    SPACY_DE = spacy.load('de')

    SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)

    train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))
    build_vocab(train_dataset, min_freq=args.min_freq)

    if args.do_train:
        main(args, train_dataset, valid_dataset, test_dataset)
    if args.do_translate:
        src, translated, attention = translate(args, test_dataset, SRC, TRG, example_idx=80)
        visualize_attention(args, src, translated, attention)

