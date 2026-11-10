import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

from evaluation import align, classification_counts, label_metrics, read, records


def main():
    cmd = argparse.ArgumentParser(description="Evaluate binary judgment macro-F1")
    cmd.add_argument("--saved-run", help="Archived dev_predictions.json with ordered gold labels, probabilities and threshold")
    cmd.add_argument("--truth")
    cmd.add_argument("--prediction")
    args = cmd.parse_args()
    if args.saved_run:
        if args.truth or args.prediction:
            cmd.error("Use --saved-run alone or supply both --truth and --prediction.")
        rows = read(args.saved_run)
        pairs = [(g, int(p >= rows["tau"])) for _, g, p in
                 zip(rows["ids"], rows["gold"], rows["prob"], strict=True)]
        if not pairs:
            raise ValueError("Empty validation run")
    else:
        if not args.truth or not args.prediction:
            cmd.error("Supply --saved-run or both --truth and --prediction.")
        truth = records(read(args.truth), "gold")
        pred = records(read(args.prediction), "pred")
        pairs = list(align(truth, pred))
    if any(type(v) is not int or v not in (0, 1) for pair in pairs for v in pair):
        raise ValueError("Expected binary integer labels")
    report = label_metrics(classification_counts(pairs), [0, 1])
    positive = report["per_label"][1]
    print(json.dumps({"macro_f1": report["macro_f1"], "precision": positive["precision"],
                      "recall": positive["recall"], "examples": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
