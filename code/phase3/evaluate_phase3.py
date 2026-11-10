import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import align, classification_counts, label_metrics, parser, read, records, unique_map


def main():
    cmd = parser("Evaluate sentence role macro-F1 across CL and IT")
    cmd.add_argument("--truth-it", required=True, help="IT references. Pass CL references with --truth.")
    cmd.add_argument("--labels", required=True)
    args = cmd.parse_args()
    def normalized(data):
        return unique_map((key.removesuffix(".txt"), value) for key, value in data.items())
    cl = normalized(records(read(args.truth), "labels"))
    it = normalized(records(read(args.truth_it), "labels"))
    truth = unique_map([*cl.items(), *it.items()])
    raw = read(args.prediction)
    if "dev_ids" in raw and "pred" in raw:
        raw = unique_map(zip(raw["dev_ids"], raw["pred"], strict=True))
    pred = normalized(raw)
    sentences = []
    for gold, output in align(truth, pred):
        sentences.extend(zip(gold, output, strict=True))
    vocab = read(args.labels)
    if isinstance(vocab, dict) and all(isinstance(v, str) for v in vocab.values()):
        vocab = vocab.values()
    report = label_metrics(classification_counts(sentences), vocab)
    report["macro_f1_all_labels"] = report.pop("macro_f1")
    report.update(cl_documents=len(cl), it_documents=len(it), sentences=len(sentences))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
