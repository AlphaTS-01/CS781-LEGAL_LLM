import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import align, counts, macro, parser, read, records


def main():
    cmd = parser("Evaluate statute macro-F1 over integer IDs 0 through the largest gold ID")
    args = cmd.parse_args()
    truth = records(read(args.truth), "gold")
    pred = records(read(args.prediction), "pred")
    gold, predicted, correct = counts()
    for actual, output in align(truth, pred):
        for values in (actual, output):
            if not isinstance(values, list) or len(set(values)) != len(values) or any(type(v) is not int or v < 0 for v in values):
                raise ValueError("Expected a list of unique nonnegative statute IDs.")
        gold.update(actual)
        predicted.update(output)
        correct.update(set(actual) & set(output))
    if not gold:
        raise ValueError("No gold statutes found")
    print(json.dumps({"macro_f1": macro((gold, predicted, correct), range(max(gold) + 1))}, indent=2))


if __name__ == "__main__":
    main()
