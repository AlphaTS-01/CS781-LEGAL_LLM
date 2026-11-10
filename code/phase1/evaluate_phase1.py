import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import align, parser, read, text_score


def main():
    cmd = parser("Evaluate continuation BERTScore F1")
    args = cmd.parse_args()
    pairs = list(align(read(args.truth), read(args.prediction)))
    refs = [gold["text"] for gold, _ in pairs]
    preds = [pred["text"] for _, pred in pairs]
    print(json.dumps({"bertscore_f1": text_score(refs, preds)}, indent=2))


if __name__ == "__main__":
    main()
