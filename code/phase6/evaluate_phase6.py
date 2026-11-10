import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import align, parser, read, records, text_score


def main():
    cmd = parser("Evaluate summary BERTScore F1")
    args = cmd.parse_args()
    truth = records(read(args.truth), "reference")
    pred = records(read(args.prediction), "summary")
    pairs = list(align(truth, pred))
    from rouge_score.rouge_scorer import RougeScorer
    scorer = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    results = [scorer.score(gold, pred) for gold, pred in pairs]
    scores = {key: sum(r[key].fmeasure for r in results) / len(pairs) for key in ("rouge1", "rougeL")}
    scores["bertscore_f1"] = text_score([g for g, _ in pairs], [p for _, p in pairs])
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
