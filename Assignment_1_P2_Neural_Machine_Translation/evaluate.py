import os
import argparse
import json
import time
import numpy as np
import pandas as pd
import sacrebleu
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer

from model import (
    lstm_decoder_vanilla,
    seq2seq_vanilla,
    seq2seq_attention,
    seq2seq_bert,
)
from training import (
    translation_dataset,
    make_collate_func,
    decode_batch,
    generate,
    compute_metrics,
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def beam_search_generate(
    model, src, src_mask, max_len, sos_id, eos_id, target_tokenizer, beam_size=5
):
    model.eval()
    batch_size = src.size(0)

    with torch.no_grad():
        if isinstance(model, seq2seq_bert):
            encoder_outputs, hidden, cell = model.encoder(src, src_mask)
        else:
            encoder_outputs, hidden, cell = model.encoder(src)

        hidden = hidden.unsqueeze(0).repeat(model.num_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(model.num_layers, 1, 1)

        encoder_outputs = (
            encoder_outputs.unsqueeze(1)
            .repeat(1, beam_size, 1, 1)
            .view(batch_size * beam_size, -1, encoder_outputs.size(-1))
        )
        src_mask = (
            src_mask.unsqueeze(1)
            .repeat(1, beam_size, 1)
            .view(batch_size * beam_size, -1)
        )

        hidden = (
            hidden.unsqueeze(2)
            .repeat(1, 1, beam_size, 1)
            .view(model.num_layers, batch_size * beam_size, -1)
        )
        cell = (
            cell.unsqueeze(2)
            .repeat(1, 1, beam_size, 1)
            .view(model.num_layers, batch_size * beam_size, -1)
        )

        scores = torch.zeros(batch_size, beam_size, device=src.device)
        scores[:, 1:] = float("-inf")

        sequences = torch.full(
            (batch_size, beam_size, 1), sos_id, dtype=torch.long, device=src.device
        )

        done = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=src.device)

        input_token = torch.full(
            (batch_size * beam_size,), sos_id, dtype=torch.long, device=src.device
        )

        for _ in range(max_len):
            if done.all():
                break

            if isinstance(model.decoder, lstm_decoder_vanilla):
                output, hidden, cell = model.decoder(
                    input_token, hidden, cell, encoder_outputs
                )
            else:
                output, hidden, cell, _ = model.decoder(
                    input_token, hidden, cell, encoder_outputs, src_mask
                )

            vocab_size = output.size(-1)
            log_probs = F.log_softmax(output, dim=-1)
            log_probs = log_probs.view(batch_size, beam_size, vocab_size)

            log_probs[done] = float("-inf")
            log_probs[done, eos_id] = 0

            current_scores = scores.unsqueeze(-1) + log_probs

            current_scores_flat = current_scores.view(batch_size, -1)
            topk_scores, topk_ids = current_scores_flat.topk(beam_size, dim=-1)

            beam_ids = topk_ids // vocab_size
            token_ids = topk_ids % vocab_size

            beam_ids_flat = (
                beam_ids
                + torch.arange(batch_size, device=src.device).unsqueeze(1) * beam_size
            ).view(-1)

            hidden = hidden[:, beam_ids_flat, :]
            cell = cell[:, beam_ids_flat, :]
            encoder_outputs_reordered = encoder_outputs[beam_ids_flat]
            src_mask_reordered = src_mask[beam_ids_flat]
            encoder_outputs = encoder_outputs_reordered
            src_mask = src_mask_reordered

            prev_seqs = sequences[
                torch.arange(batch_size, device=src.device).unsqueeze(1), beam_ids
            ]
            sequences = torch.cat([prev_seqs, token_ids.unsqueeze(-1)], dim=-1)

            done = done[
                torch.arange(batch_size, device=src.device).unsqueeze(1), beam_ids
            ]
            done = done | (token_ids == eos_id)
            scores = topk_scores

            input_token = token_ids.view(batch_size * beam_size)

        seq_lens = (sequences != eos_id).sum(dim=-1).float().clamp(min=1)
        norm_scores = scores / seq_lens
        best_beam_ids = norm_scores.argmax(dim=-1)

        best_sequences = sequences[
            torch.arange(batch_size, device=src.device), best_beam_ids
        ]

        result = best_sequences[:, 1:]
        eos_mask = result == eos_id
        first_eos = eos_mask.float().argmax(dim=-1)
        has_eos = eos_mask.any(dim=-1)
        for i in range(batch_size):
            if has_eos[i]:
                result[i, first_eos[i] :] = target_tokenizer.pad_token_id

        return result


def build_model(model_type, output_vocab_size, src_word2idx=None):
    EMBED_DIM = 100
    HIDDEN_DIM = 256
    NUM_LAYERS = 1

    if model_type == 1:
        model = seq2seq_vanilla(
            input_vocab_size=len(src_word2idx),
            output_vocab_size=output_vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            pretrained_embeddings=torch.zeros(len(src_word2idx), EMBED_DIM),
        )
    elif model_type == 2:
        model = seq2seq_attention(
            input_vocab_size=len(src_word2idx),
            output_vocab_size=output_vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            pretrained_embeddings=torch.zeros(len(src_word2idx), EMBED_DIM),
        )
    elif model_type in (3, 4):
        model = seq2seq_bert(
            output_vocab_size=output_vocab_size,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            freeze_bert=(model_type == 3),
        )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--model_type", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument(
        "--language", type=str, required=True, choices=["Hindi", "Marathi"]
    )
    args = parser.parse_args()

    language = args.language

    exp_names = {
        1: "vanilla_glove",
        2: "attention_glove",
        3: "bert_frozen",
        4: "bert_unfrozen",
    }
    exp = exp_names[args.model_type]
    model_path = os.path.join(args.save_dir, language, f"{exp}.pth")

    checkpoint = torch.load(model_path, map_location=device)

    encoder_type = checkpoint["encoder_type"]

    target_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(checkpoint["target_tokenizer_json"]),
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<sos>",
        eos_token="<eos>",
    )

    if encoder_type == "glove":
        src_word2idx = checkpoint["src_word2idx"]
        src_idx2word = checkpoint["src_idx2word"]
        src_pad_idx = src_word2idx["<pad>"]
        bert_src_tokenizer = None
    elif encoder_type == "bert":
        bert_src_tokenizer = BertTokenizer.from_pretrained(
            checkpoint["bert_model_name"]
        )
        src_pad_idx = bert_src_tokenizer.pad_token_id
        src_word2idx = None
        src_idx2word = None

    trg_pad_idx = target_tokenizer.pad_token_id
    output_vocab_size = len(target_tokenizer)

    test_dataframe = pd.read_csv(args.test_path)

    def extract_english(text):
        prefix = f"translate to {language}: "
        return text[len(prefix) :] if text.startswith(prefix) else text

    test_src = test_dataframe["source"].apply(extract_english).tolist()
    test_trg = test_dataframe["translated"].tolist()

    test_dataset = translation_dataset(
        test_src,
        test_trg,
        encoder_type,
        src_word2idx=src_word2idx if encoder_type == "glove" else None,
        bert_tokenizer=bert_src_tokenizer if encoder_type == "bert" else None,
        target_tokenizer=target_tokenizer,
        max_length=512,
    )

    collate_fn = make_collate_func(src_pad_idx, trg_pad_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = build_model(args.model_type, output_vocab_size, src_word2idx).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {model_path}\n")

    beam_sizes = [1, 5, 10, 20]
    beam_results = {}

    for beam_size in beam_sizes:
        all_preds, all_refs, all_srcs = [], [], []
        total_time = 0.0
        total_sents = 0

        with torch.no_grad():
            for src, trg, src_mask in tqdm(test_loader):
                src, trg, src_mask = (
                    src.to(device),
                    trg.to(device),
                    src_mask.to(device),
                )
                trg_out = trg[:, 1:]
                batch_sz = src.size(0)

                t_start = time.perf_counter()

                if beam_size == 1:
                    generated = generate(
                        model,
                        src,
                        src_mask,
                        max_len=trg.size(1),
                        sos_id=target_tokenizer.bos_token_id,
                        eos_id=target_tokenizer.eos_token_id,
                    )
                else:
                    generated = beam_search_generate(
                        model,
                        src,
                        src_mask,
                        max_len=trg.size(1),
                        sos_id=target_tokenizer.bos_token_id,
                        eos_id=target_tokenizer.eos_token_id,
                        target_tokenizer=target_tokenizer,
                        beam_size=beam_size,
                    )

                t_end = time.perf_counter()
                total_time += t_end - t_start
                total_sents += batch_sz

                all_preds.extend(decode_batch(generated, target_tokenizer))
                all_refs.extend(decode_batch(trg_out, target_tokenizer))

                if encoder_type == "bert":
                    all_srcs.extend(decode_batch(src, bert_src_tokenizer))
                else:
                    all_srcs.extend(
                        " ".join(
                            src_idx2word.get(t.item(), "<unk>")
                            for t in row
                            if t.item()
                            not in (
                                src_pad_idx,
                                src_word2idx["<sos>"],
                                src_word2idx["<eos>"],
                            )
                        )
                        for row in src.cpu()
                    )

        scores = compute_metrics(all_preds, all_refs)
        avg_time_per_sent = total_time / total_sents

        beam_results[beam_size] = {
            **scores,
            "avg_time_per_sent_sec": round(avg_time_per_sent, 5),
        }

        print(
            f"\n=== Beam Size {beam_size} ({'Greedy' if beam_size == 1 else 'Beam Search'}) ===\n"
            f"  BLEU : {scores['bleu']:.2f}\n"
            f"  chrF : {scores['chrf']:.2f}\n"
            f"  TER  : {scores['ter']:.2f}\n"
            f"  Avg decoding time / sentence : {avg_time_per_sent:.5f}s\n"
        )

        sent_scores = [
            (
                sacrebleu.sentence_bleu(pred, [ref]).score,
                src_sent,
                pred,
                ref,
            )
            for src_sent, pred, ref in zip(all_srcs, all_preds, all_refs)
        ]
        sent_scores.sort(key=lambda x: x[0], reverse=True)

        print("  ── Top 5 Translations (highest sentence BLEU) ──")
        for rank, (bleu_s, src_sent, pred, ref) in enumerate(sent_scores[:5], 1):
            print(f"  [{rank}] BLEU: {bleu_s:.2f}")
            print(f"       SRC : {src_sent}")
            print(f"       REF : {ref}")
            print(f"       PRED: {pred}")

        print("\n  ── Bottom 5 Translations (lowest sentence BLEU) ──")
        for rank, (bleu_s, src_sent, pred, ref) in enumerate(sent_scores[-5:][::-1], 1):
            print(f"  [{rank}] BLEU: {bleu_s:.2f}")
            print(f"       SRC : {src_sent}")
            print(f"       REF : {ref}")
            print(f"       PRED: {pred}")

        random_indices = np.random.choice(
            len(sent_scores), size=min(5, len(sent_scores)), replace=False
        )
        random_samples = [sent_scores[i] for i in random_indices]
        print("\n  ── 5 Random Translations ──")
        for rank, (bleu_s, src_sent, pred, ref) in enumerate(random_samples, 1):
            print(f"  [{rank}] BLEU: {bleu_s:.2f}")
            print(f"       SRC : {src_sent}")
            print(f"       REF : {ref}")
            print(f"       PRED: {pred}")

    out_path = os.path.join(args.save_dir, language, f"beam_search_results_{exp}.json")
    with open(out_path, "w") as f:
        json.dump(beam_results, f, indent=4)
    print(f"\nResults saved to {out_path}")
