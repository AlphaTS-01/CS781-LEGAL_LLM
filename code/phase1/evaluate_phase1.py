import json
import math
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import align, parser, read, text_score


def lexical_scores(refs, preds):
    """Whitespace-token ROUGE-L and unsmoothed corpus BLEU-4."""
    matches = [0] * 4
    totals = [0] * 4
    gold_length = predicted_length = 0
    rouge = 0.0
    for reference, prediction in zip(refs, preds, strict=True):
        gold, output = reference.split(), prediction.split()
        gold_length += len(gold)
        predicted_length += len(output)
        previous = [0] * (len(gold) + 1)
        for word in output:
            current = [0]
            for j, target in enumerate(gold, 1):
                current.append(previous[j - 1] + 1 if word == target else max(previous[j], current[-1]))
            previous = current
        rouge += 2 * previous[-1] / max(1, len(output) + len(gold))
        for n in range(1, 5):
            actual = Counter(zip(*(gold[j:] for j in range(n))))
            predicted = Counter(zip(*(output[j:] for j in range(n))))
            matches[n - 1] += sum((actual & predicted).values())
            totals[n - 1] += sum(predicted.values())
    bleu = 0.0
    if all(matches):
        mean_log = sum(math.log(h / t) for h, t in zip(matches, totals)) / 4
        bleu = math.exp(min(0, 1 - gold_length / predicted_length) + mean_log)
    return {"rougeL": rouge / len(refs), "bleu4": bleu}


def main():
    cmd = parser("Evaluate continuation BERTScore F1")
    args = cmd.parse_args()
    pairs = list(align(read(args.truth), read(args.prediction)))
    refs = [gold["text"] for gold, _ in pairs]
    preds = [pred["text"] for _, pred in pairs]
    scores = {"bertscore_f1": text_score(refs, preds)}
    scores.update(lexical_scores(refs, preds))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
