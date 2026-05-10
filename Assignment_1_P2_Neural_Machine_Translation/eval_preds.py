import argparse
import pandas as pd
import sacrebleu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="CSV with 'translated' column (ground truth)",
    )
    parser.add_argument(
        "--generated",
        type=str,
        required=True,
        help="CSV with 'translated' column (predictions)",
    )
    args = parser.parse_args()

    refs = pd.read_csv(args.source)["translated"].astype(str).tolist()
    preds = pd.read_csv(args.generated)["translated"].astype(str).tolist()

    assert len(refs) == len(
        preds
    ), f"Length mismatch: {len(refs)} refs vs {len(preds)} preds"

    bleu = sacrebleu.corpus_bleu(preds, [refs], force=True).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    ter = sacrebleu.corpus_ter(preds, [refs]).score

    print(f"BLEU : {bleu:.2f}")
    print(f"chrF : {chrf:.2f}")
    print(f"TER  : {ter:.2f}")
