import os
import spacy
from models.decoder import *
from models.encoder import *
from models.transformer import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchtext.data.metrics import bleu_score


def translate(args, src, SRC, TRG, max_len=50, logging=True):
    device = torch.device('cpu')

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
    model.load_state_dict(
        torch.load(
            os.path.join(args.save_model_path, "translate.pt"), map_location=device
        )
    )
    model.eval()

    if isinstance(src, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower for token in nlp(src)]
    else:
        tokens = [token.lower() for token in src]

    tokens = [SRC.init_token] + tokens + [SRC.eos_token]

    src_indexes = [SRC.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.get_input_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [TRG.vocab.stoi[TRG.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.get_output_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)  # 출력 문장에 더하기

        if pred_token == TRG.vocab.stoi[TRG.eos_token]:
            break
    trg_tokens = [TRG.vocab.itos[i] for i in trg_indexes]
    translated_tokens = trg_tokens[1:]

    if logging:
        print("Translated tokens:", translated_tokens)

    return src, translated_tokens, attention


def translate_example(args, test_dataset, SRC, TRG, example_idx):
    print("==== Translate Examples ====")
    src = vars(test_dataset.examples[example_idx])['src']
    trg = vars(test_dataset.examples[example_idx])['trg']

    print(f'Source tokens: {src}')
    print(f'Target tokens: {trg}')
    return translate(args, src, SRC, TRG)


def visualize_attention(args, src, translated, attention, n_heads=8, n_rows=4, n_cols=2):
    assert n_rows * n_cols == n_heads
    assert args.n_heads == n_heads
    print("==== Visualize attention Examples ====")

    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        attention_ = attention.squeeze(0)[i].cpu().detach().numpy()
        cax = ax.matshow(attention_, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in src] + ['<eos>'], rotation=45)
        ax.set_yticklabels([''] + translated)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


def check_bleu_score(args, dataset, SRC, TRG, max_len=50):
    trgs = []
    pred_trgs = []
    index = 0
    for data in dataset:
        src = vars(data)['src']
        trg = vars(data)['trg']

        _, pred_trg, _ = translate(args, src, SRC, TRG, logging=False)
        pred_trg = pred_trg[:-1]
        pred_trgs.append(pred_trg)
        trgs.append([trg])

        index += 1
        if (index+1) % 100 ==0:
            print(f"[{index+1}/{len(dataset)}]")
            print(f"Predict: {pred_trg}")
            print(f"Anser: {trg}")

    belu_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
    print(f"Total BLEU Score = {belu_score*100:.2f}")

    individual_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
    individual_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 1, 0, 0])
    individual_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 0, 1, 0])
    individual_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 0, 0, 1])

    print(f'Individual BLEU1 score = {individual_bleu1_score * 100:.2f}')
    print(f'Individual BLEU2 score = {individual_bleu2_score * 100:.2f}')
    print(f'Individual BLEU3 score = {individual_bleu3_score * 100:.2f}')
    print(f'Individual BLEU4 score = {individual_bleu4_score * 100:.2f}')

    cumulative_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
    cumulative_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/2, 1/2, 0, 0])
    cumulative_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/3, 1/3, 1/3, 0])
    cumulative_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/4, 1/4, 1/4, 1/4])

    print(f'Cumulative BLEU1 score = {cumulative_bleu1_score * 100:.2f}')
    print(f'Cumulative BLEU2 score = {cumulative_bleu2_score * 100:.2f}')
    print(f'Cumulative BLEU3 score = {cumulative_bleu3_score * 100:.2f}')
    print(f'Cumulative BLEU4 score = {cumulative_bleu4_score * 100:.2f}')
