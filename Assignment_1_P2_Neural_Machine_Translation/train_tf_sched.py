import os
import json
import re
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sacrebleu
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchtext.vocab as vocab
from transformers import BertTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from model import (
    lstm_decoder_vanilla,
    seq2seq_vanilla,
    seq2seq_attention,
    seq2seq_bert,
)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(device)


class translation_dataset(Dataset):
    def __init__(
        self,
        src_sentences,
        target_sentences,
        encoder_type,
        src_word2idx=None,
        bert_tokenizer=None,
        target_tokenizer=None,
        max_length=512,
    ):
        self.encoder_type = encoder_type

        if encoder_type == "glove":
            self.tokenized_src = [
                torch.tensor(tokenize_source_glove(s, src_word2idx), dtype=torch.long)
                for s in src_sentences
            ]
            self.src_attention_masks = None

        elif encoder_type == "bert":
            encoded = bert_tokenizer(
                src_sentences,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
            self.tokenized_src = [
                torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]
            ]
            self.src_attention_masks = [
                torch.tensor(mask, dtype=torch.long)
                for mask in encoded["attention_mask"]
            ]

        self.tokenized_trg = [
            torch.tensor(tokenize_target(s, target_tokenizer), dtype=torch.long)
            for s in target_sentences
        ]

    def __len__(self):
        return len(self.tokenized_src)

    def __getitem__(self, idx):
        src = self.tokenized_src[idx]
        trg = self.tokenized_trg[idx]

        mask = (
            self.src_attention_masks[idx]
            if self.src_attention_masks is not None
            else None
        )
        return src, trg, mask


def build_source_glove(corpus):
    counter = Counter()

    for sentence in corpus:
        counter.update(TOKEN_PATTERN.findall(sentence.lower()))

    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
    all_words = special_tokens + [w for w, c in counter.items() if c >= MIN_SOURCE_FREQ]

    word2idx = {w: i for i, w in enumerate(all_words)}
    idx2word = {i: w for w, i in word2idx.items()}

    glove = vocab.GloVe(name="6B", dim=GLOVE_DIM)
    pretrained_glove_embeds = np.zeros((len(word2idx), GLOVE_DIM))

    for word, idx in word2idx.items():
        if word in glove.stoi:
            pretrained_glove_embeds[idx] = glove.vectors[glove.stoi[word]].numpy()
        else:
            pretrained_glove_embeds[idx] = np.random.normal(0, 0.1, GLOVE_DIM)

    pretrained_glove_embeds = torch.tensor(pretrained_glove_embeds, dtype=torch.float32)

    print(f"Source Glove Embeddings initialised with vocab size of {len(word2idx)}")
    return word2idx, idx2word, pretrained_glove_embeds


def train_target_bpe(corpus, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"],
        vocab_size=vocab_size,
        min_frequency=MIN_TARGET_FREQ,
    )

    tokenizer.train_from_iterator(corpus, trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<sos>",
        eos_token="<eos>",
    )

    print(f"Target BPE trained with vocab size of {len(fast_tokenizer)}")

    return fast_tokenizer


def tokenize_source_glove(sentence, word2idx):
    sos, eos, unk = word2idx["<sos>"], word2idx["<eos>"], word2idx["<unk>"]
    tokens = TOKEN_PATTERN.findall(sentence.lower())
    ids = [word2idx.get(tok, unk) for tok in tokens]
    return [sos] + ids + [eos]


def tokenize_target(sentence, tokenizer):
    ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
    sos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    return [sos] + ids + [eos]


def make_collate_func(src_pad_idx, trg_pad_idx):
    def collate_fn(batch):
        src_batch, trg_batch, mask_batch = zip(*batch)

        src_padded = pad_sequence(
            src_batch, batch_first=True, padding_value=src_pad_idx
        )
        trg_padded = pad_sequence(
            trg_batch, batch_first=True, padding_value=trg_pad_idx
        )

        if mask_batch[0] is not None:
            src_mask = pad_sequence(mask_batch, batch_first=True, padding_value=0)
        else:
            src_mask = (src_padded != src_pad_idx).long()

        return src_padded, trg_padded, src_mask

    return collate_fn


def create_dataloaders(encoder_type):
    if encoder_type == "glove":
        train_dataset = translation_dataset(
            train_src,
            train_trg,
            encoder_type,
            src_word2idx=src_word2idx,
            target_tokenizer=target_tokenizer,
        )
        val_dataset = translation_dataset(
            val_src,
            val_trg,
            encoder_type,
            src_word2idx=src_word2idx,
            target_tokenizer=target_tokenizer,
        )
        test_dataset = translation_dataset(
            test_src,
            test_trg,
            encoder_type,
            src_word2idx=src_word2idx,
            target_tokenizer=target_tokenizer,
        )
        collate_fn = make_collate_func(src_glove_pad_idx, trg_pad_idx)

    elif encoder_type == "bert":
        train_dataset = translation_dataset(
            train_src,
            train_trg,
            encoder_type,
            bert_tokenizer=bert_src_tokenizer,
            max_length=512,
            target_tokenizer=target_tokenizer,
        )
        val_dataset = translation_dataset(
            val_src,
            val_trg,
            encoder_type,
            bert_tokenizer=bert_src_tokenizer,
            max_length=512,
            target_tokenizer=target_tokenizer,
        )
        test_dataset = translation_dataset(
            test_src,
            test_trg,
            encoder_type,
            bert_tokenizer=bert_src_tokenizer,
            max_length=512,
            target_tokenizer=target_tokenizer,
        )
        collate_fn = make_collate_func(src_bert_pad_idx, trg_pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def decode_batch(token_ids_batch, tokenizer):
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    if isinstance(token_ids_batch, torch.Tensor):
        token_ids_batch = token_ids_batch.cpu().tolist()

    results = []
    for seq in token_ids_batch:
        cleaned = []

        for tok in seq:
            if tok == eos_id:
                break
            if tok != pad_id:
                cleaned.append(tok)

        results.append(tokenizer.decode(cleaned, skip_special_tokens=True))

    return results


def plot_metrics(epochs_range, train_vals, val_vals, ylabel, title):
    plt.plot(epochs_range, train_vals, label="Train")
    plt.plot(epochs_range, val_vals, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()


def round_metrics(metrics, decimals=4):
    def round_value(value):
        if isinstance(value, float):
            return round(value, decimals)
        elif isinstance(value, list):
            return [round_value(v) for v in value]
        elif isinstance(value, tuple):
            return tuple(round_value(v) for v in value)
        elif isinstance(value, dict):
            return {k: round_value(v) for k, v in value.items()}
        return value

    return {key: round_value(values) for key, values in metrics.items()}


def compute_metrics(predictions, references):
    bleu = sacrebleu.corpus_bleu(predictions, [references], force=True).score
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score
    ter = sacrebleu.corpus_ter(predictions, [references]).score

    return {
        "bleu": bleu,
        "chrf": chrf,
        "ter": ter,
    }


def generate(model, src, src_mask, max_len, sos_id, eos_id):
    model.eval()
    batch_size = src.size(0)

    if isinstance(model, seq2seq_bert):
        encoder_outputs, hidden, cell = model.encoder(src, src_mask)
    else:
        encoder_outputs, hidden, cell = model.encoder(src)

    hidden = hidden.unsqueeze(0).repeat(model.num_layers, 1, 1)
    cell = cell.unsqueeze(0).repeat(model.num_layers, 1, 1)

    input_token = torch.full((batch_size,), sos_id, device=src.device)

    outputs = []
    finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

    for _ in range(max_len):
        if isinstance(model.decoder, lstm_decoder_vanilla):
            output, hidden, cell = model.decoder(
                input_token, hidden, cell, encoder_outputs
            )
        else:
            output, hidden, cell, _ = model.decoder(
                input_token, hidden, cell, encoder_outputs, src_mask
            )

        next_token = output.argmax(dim=1)
        outputs.append(next_token)

        finished |= next_token == eos_id

        if finished.all():
            break

        input_token = next_token

    outputs = torch.stack(outputs, dim=1)
    return outputs


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    exp,
    target_tokenizer,
):
    metrics = {
        "losses": ([], []),
        "bleu": [],
        "chrf": [],
        "ter": [],
        "tf_probs": [],
    }

    best_val_bleu = 0.0

    for epoch in range(1, epochs + 1):

        # ── TRAINING ────────────────────────────────────────────────────────
        running_train_loss = 0.0
        model.train()
        tf_ratio = 8 / (8 + np.exp(epoch / 8))
        metrics["tf_probs"].append(tf_ratio)

        for src, trg, src_mask in tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs} - Training"
        ):
            src, trg, src_mask = src.to(device), trg.to(device), src_mask.to(device)
            optimizer.zero_grad()

            if isinstance(model, seq2seq_bert):
                outputs = model(src, src_mask, trg, teacher_forcing_ratio=tf_ratio)
            elif isinstance(model, seq2seq_attention):
                outputs = model(
                    src, trg, teacher_forcing_ratio=tf_ratio, src_mask=src_mask
                )
            else:
                outputs = model(src, trg, teacher_forcing_ratio=tf_ratio)

            trg_out = trg[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), trg_out.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            running_train_loss += loss.item()

        scheduler.step()

        torch.cuda.empty_cache()

        # ── VALIDATION ──────────────────────────────────────────────────────
        model.eval()
        running_val_loss = 0.0
        val_preds, val_refs = [], []

        with torch.no_grad():
            for src, trg, src_mask in tqdm(
                val_loader, desc=f"Epoch {epoch}/{epochs} - Validation", leave=False
            ):
                src, trg, src_mask = src.to(device), trg.to(device), src_mask.to(device)

                if isinstance(model, seq2seq_bert):
                    outputs = model(src, src_mask, trg, teacher_forcing_ratio=0.0)
                elif isinstance(model, seq2seq_attention):
                    outputs = model(
                        src, trg, teacher_forcing_ratio=0.0, src_mask=src_mask
                    )
                else:
                    outputs = model(src, trg, teacher_forcing_ratio=0.0)

                trg_out = trg[:, 1:]
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)), trg_out.reshape(-1)
                )
                running_val_loss += loss.item()

                val_preds.extend(
                    decode_batch(outputs.argmax(dim=-1).cpu(), target_tokenizer)
                )
                val_refs.extend(decode_batch(trg_out.cpu(), target_tokenizer))

        val_metrics = compute_metrics(val_preds, val_refs)

        running_train_loss /= len(train_loader)
        running_val_loss /= len(val_loader)

        metrics["losses"][0].append(running_train_loss)
        metrics["losses"][1].append(running_val_loss)

        for key in ["bleu", "chrf", "ter"]:
            metrics[key].append(val_metrics[key])

        print(
            f"Epoch [{epoch}/{epochs}]:\n"
            f"  Train  | Loss: {metrics['losses'][0][-1]:.4f}\n"
            f"  Val    | Loss: {metrics['losses'][1][-1]:.4f}  "
            f"BLEU: {metrics['bleu'][-1]:.2f}  "
            f"chrF: {metrics['chrf'][-1]:.2f}  "
            f"TER: {metrics['ter'][-1]:.2f}\n"
        )

        val_bleu = metrics["bleu"][-1]

        if val_bleu >= best_val_bleu:
            best_val_bleu = val_bleu
            print(f"Best model saving at epoch {epoch} with Val BLEU: {val_bleu:.2f}\n")

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_type": encoder_type,
                    "src_word2idx": src_word2idx if encoder_type == "glove" else None,
                    "src_idx2word": src_idx2word if encoder_type == "glove" else None,
                    "target_tokenizer_json": target_tokenizer.backend_tokenizer.to_str(),
                    "bert_model_name": (
                        "bert-base-cased" if encoder_type == "bert" else None
                    ),
                },
                os.path.join(save_dir, f"{exp}_tf_sched.pth"),
            )

        torch.cuda.empty_cache()

    checkpoint = torch.load(
        os.path.join(save_dir, f"{exp}_tf_sched.pth"), map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\nLoaded best checkpoint from {exp}_tf_sched.pth\n")

    for split_name, loader in [
        ("Train", train_loader),
        ("Val", val_loader),
        ("Test", test_loader),
    ]:
        split_preds, split_refs = [], []
        split_loss = 0.0

        with torch.no_grad():
            for src, trg, src_mask in tqdm(
                loader, desc=f"Final Eval - {split_name}", leave=False
            ):
                src, trg, src_mask = src.to(device), trg.to(device), src_mask.to(device)

                if isinstance(model, seq2seq_bert):
                    outputs = model(src, src_mask, trg, teacher_forcing_ratio=0.0)
                elif isinstance(model, seq2seq_attention):
                    outputs = model(
                        src, trg, teacher_forcing_ratio=0.0, src_mask=src_mask
                    )
                else:
                    outputs = model(src, trg, teacher_forcing_ratio=0.0)

                trg_out = trg[:, 1:]
                split_loss += criterion(
                    outputs.reshape(-1, outputs.size(-1)), trg_out.reshape(-1)
                ).item()

                generated = generate(
                    model,
                    src,
                    src_mask,
                    max_len=trg.size(1),
                    sos_id=target_tokenizer.bos_token_id,
                    eos_id=target_tokenizer.eos_token_id,
                )

                split_preds.extend(decode_batch(generated, target_tokenizer))
                split_refs.extend(decode_batch(trg_out, target_tokenizer))

        split_loss /= len(loader)
        split_metrics = compute_metrics(split_preds, split_refs)

        print(
            f"=== Final {split_name} Metrics (Best Model) ===\n"
            f"  Loss : {split_loss:.4f}\n"
            f"  BLEU : {split_metrics['bleu']:.2f}\n"
            f"  chrF : {split_metrics['chrf']:.2f}\n"
            f"  TER  : {split_metrics['ter']:.2f}\n"
        )

    return metrics


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+|_|[^\w\s]")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument(
        "--language", type=str, required=True, choices=["Hindi", "Marathi"]
    )
    parser.add_argument("--model_type", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    language = args.language
    model_type = args.model_type

    save_dir = os.path.join(args.save_dir, language)
    os.makedirs(save_dir, exist_ok=True)

    # All Hyperparameters

    GLOVE_DIM = 100
    MIN_SOURCE_FREQ = 1
    MIN_TARGET_FREQ = 1
    TARGET_VOCAB_SIZE = 10000
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 64

    EMBED_DIM = 100
    HIDDEN_DIM = 256
    NUM_LAYERS = 1
    EPOCHS = 30
    LR = 1e-3

    train_dataframe = pd.read_csv(args.train_path)
    val_dataframe = pd.read_csv(args.val_path)
    test_dataframe = pd.read_csv(args.test_path)

    def extract_english(text):
        prefix = f"translate to {language}: "

        return text[len(prefix) :] if text.startswith(prefix) else text

    train_src = train_dataframe["source"].apply(extract_english).tolist()
    train_trg = train_dataframe["translated"].tolist()

    val_src = val_dataframe["source"].apply(extract_english).tolist()
    val_trg = val_dataframe["translated"].tolist()

    test_src = test_dataframe["source"].apply(extract_english).tolist()
    test_trg = test_dataframe["translated"].tolist()

    src_word2idx, src_idx2word, src_pretrained_glove_embeds = build_source_glove(
        train_src
    )
    bert_src_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    target_tokenizer = train_target_bpe(train_trg, vocab_size=TARGET_VOCAB_SIZE)
    output_vocab_size = len(target_tokenizer)

    src_glove_pad_idx = src_word2idx["<pad>"]
    src_bert_pad_idx = bert_src_tokenizer.pad_token_id
    trg_pad_idx = target_tokenizer.pad_token_id

    if model_type == 1:
        encoder_type = "glove"
        train_loader, val_loader, test_loader = create_dataloaders(encoder_type)
        input_vocab_size = len(src_word2idx)

        model = seq2seq_vanilla(
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            pretrained_embeddings=src_pretrained_glove_embeds,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-4
        )

    elif model_type == 2:
        encoder_type = "glove"
        train_loader, val_loader, test_loader = create_dataloaders(encoder_type)
        input_vocab_size = len(src_word2idx)

        model = seq2seq_attention(
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            pretrained_embeddings=src_pretrained_glove_embeds,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-4
        )

    elif model_type == 3:
        encoder_type = "bert"
        train_loader, val_loader, test_loader = create_dataloaders(encoder_type)
        input_vocab_size = len(src_word2idx)

        model = seq2seq_bert(
            output_vocab_size=output_vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            freeze_bert=True,
        ).to(device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LR
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-4
        )

    elif model_type == 4:
        encoder_type = "bert"
        train_loader, val_loader, test_loader = create_dataloaders(encoder_type)
        input_vocab_size = len(src_word2idx)

        model = seq2seq_bert(
            output_vocab_size=output_vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            freeze_bert=False,
        ).to(device)

        bert_params = list(model.encoder.bert.parameters())
        non_bert_params = [
            p for p in model.parameters() if not any(p is bp for bp in bert_params)
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": bert_params, "lr": 1e-4},
                {"params": non_bert_params, "lr": LR},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-4
        )

    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    exp_names = {
        1: "vanilla_glove",
        2: "attention_glove",
        3: "bert_frozen",
        4: "bert_unfrozen",
    }
    exp = exp_names[model_type]

    metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        exp=exp,
        target_tokenizer=target_tokenizer,
    )

    with open(os.path.join(save_dir, f"metrics_{exp}.json"), "w") as f:
        json.dump(round_metrics(metrics), f, indent=4)

    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    fig.suptitle(f"Training Curves — {exp} (Inverse Sigmoid TF Schedule)", fontsize=14)

    tf_probs = metrics["tf_probs"]

    for ax, (loss_vals, split_name) in zip(
        axes[0],
        [
            (metrics["losses"][0], "Train"),
            (metrics["losses"][1], "Validation"),
        ],
    ):
        ax2 = ax.twinx()
        ax.plot(epochs_range, loss_vals, color="steelblue", label=f"{split_name} Loss")
        ax2.plot(
            epochs_range,
            tf_probs,
            color="darkorange",
            linestyle="--",
            label="TF Probability",
        )
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss", color="steelblue")
        ax2.set_ylabel("Teacher Forcing Probability", color="darkorange")
        ax.set_title(f"{split_name} Loss vs TF Schedule")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax.grid()

    axes[0, 2].set_visible(False)

    for ax, (key, ylabel, title) in zip(
        axes[1],
        [
            ("bleu", "BLEU Score", "BLEU"),
            ("chrf", "chrF Score", "chrF"),
            ("ter", "TER Score", "TER"),
        ],
    ):
        ax.plot(epochs_range, metrics[key], label="Validation")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"metrics_{exp}.png"), dpi=150)
    plt.close()
