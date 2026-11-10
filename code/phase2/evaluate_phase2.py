import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import align, counts, label_metrics, parser, read


def main():
    cmd = parser("Evaluate exact entity span and label matches")
    cmd.add_argument("--labels", required=True, help="One entity label per line")
    args = cmd.parse_args()
    labels = [s.strip() for s in Path(args.labels).read_text().splitlines() if s.strip() and s.strip() != "O"]
    gold, predicted, correct = counts()
    for actual, output in align(read(args.truth), read(args.prediction)):
        spans = []
        for entries in (actual, output):
            values = set()
            for start, end, label in entries:
                if type(start) is not int or type(end) is not int or start < 0 or start >= end or label not in labels:
                    raise ValueError("Invalid entity span or label")
                values.add((start, end, label))
            spans.append(values)
        gold.update(span[2] for span in spans[0])
        predicted.update(span[2] for span in spans[1])
        correct.update(span[2] for span in spans[0] & spans[1])
    report = label_metrics((gold, predicted, correct), labels)
    report["strict_macro_f1"] = report.pop("macro_f1")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
