import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer

from training import translation_dataset, make_collate_func, decode_batch, generate
from evaluate import beam_search_generate, build_model

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--language", type=str, default="Hindi", choices=["Hindi", "Marathi"]
    )
    parser.add_argument("--mode", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument(
        "--decoding_strategy", type=str, default="greedy", choices=["greedy", "beam"]
    )
    args = parser.parse_args()

    exp_names = {
        1: "vanilla_glove",
        2: "attention_glove",
        3: "bert_frozen",
        4: "bert_unfrozen",
    }
    exp = exp_names[args.mode]
    model_path = os.path.join(args.checkpoint, f"{exp}.pth")

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
    else:
        bert_src_tokenizer = BertTokenizer.from_pretrained(
            checkpoint["bert_model_name"]
        )
        src_pad_idx = bert_src_tokenizer.pad_token_id
        src_word2idx = None
        src_idx2word = None

    trg_pad_idx = target_tokenizer.pad_token_id
    output_vocab_size = len(target_tokenizer)

    df = pd.read_csv(args.input)

    def extract_english(text):
        prefix = f"translate to {args.language}: "
        return text[len(prefix) :] if text.startswith(prefix) else text

    src_sentences = df["source"].apply(extract_english).tolist()
    dummy_trg = [""] * len(src_sentences)

    dataset = translation_dataset(
        src_sentences,
        dummy_trg,
        encoder_type,
        src_word2idx=src_word2idx if encoder_type == "glove" else None,
        bert_tokenizer=bert_src_tokenizer if encoder_type == "bert" else None,
        target_tokenizer=target_tokenizer,
        max_length=512,
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=make_collate_func(src_pad_idx, trg_pad_idx),
        pin_memory=True,
    )

    model = build_model(args.mode, output_vocab_size, src_word2idx).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {model_path}")
    print(f"Decoding strategy : {args.decoding_strategy}\n")

    all_preds = []

    with torch.no_grad():
        for src, trg, src_mask in loader:
            src, trg, src_mask = src.to(device), trg.to(device), src_mask.to(device)

            if args.decoding_strategy == "greedy":
                generated = generate(
                    model,
                    src,
                    src_mask,
                    max_len=512,
                    sos_id=target_tokenizer.bos_token_id,
                    eos_id=target_tokenizer.eos_token_id,
                )
            else:
                generated = beam_search_generate(
                    model,
                    src,
                    src_mask,
                    max_len=512,
                    sos_id=target_tokenizer.bos_token_id,
                    eos_id=target_tokenizer.eos_token_id,
                    target_tokenizer=target_tokenizer,
                    beam_size=5,
                )

            all_preds.extend(decode_batch(generated, target_tokenizer))

    out_df = pd.DataFrame({"source": df["source"], "translated": all_preds})
    out_df.to_csv(args.output, index=False)
    print(f"Translations saved to {args.output}")
