import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import align, fscore, parser, read, records


def main():
    cmd = parser("Evaluate micro precision, recall and F1 at seven")
    cmd.add_argument("--candidate-ids", required=True, help="JSON list of candidate IDs or mapping keyed by candidate ID")
    args = cmd.parse_args()
    truth = records(read(args.truth), "relevant_candidates")
    pred = read(args.prediction)
    if "identity" in pred and "predictions" in pred:
        pred = pred["predictions"]
    candidates = set(read(args.candidate_ids))
    hits = gold_count = predicted_count = 0
    for actual, output in align(truth, pred):
        if isinstance(actual, dict):
            actual = actual["relevant_candidates"]
        if not isinstance(actual, list) or not isinstance(output, list):
            raise ValueError("Expected lists of case IDs")
        if len(output) != 7 or len(set(output)) != 7 or set(output) - candidates:
            raise ValueError("Each ranking needs seven unique, known candidate IDs.")
        relevant = set(actual)
        hits += len(relevant.intersection(output))
        gold_count += len(relevant)
        predicted_count += len(output)
    print(json.dumps({"micro_f1_at7": fscore(hits, gold_count, predicted_count),
                      "micro_precision_at7": hits / predicted_count,
                      "micro_recall_at7": hits / gold_count if gold_count else 0}, indent=2))


if __name__ == "__main__":
    main()
