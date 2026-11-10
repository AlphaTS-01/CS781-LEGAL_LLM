"""Common input checks and metrics for the capstone's local evaluators."""
import argparse
import json
from collections import Counter
from pathlib import Path


def unique_map(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key: {key}")
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_map)


def parser(description):
    command = argparse.ArgumentParser(description=description)
    command.add_argument("--truth", required=True)
    command.add_argument("--prediction", required=True)
    return command


def records(data, field):
    if isinstance(data, list):
        return unique_map((str(row["id"]), row[field]) for row in data)
    return data


def align(truth, predictions):
    if not isinstance(truth, dict) or not truth or truth.keys() != predictions.keys():
        raise ValueError("Prediction IDs must match all reference IDs exactly.")
    return ((truth[key], predictions[key]) for key in truth)


def fscore(correct, gold, predicted):
    return 2 * correct / (gold + predicted) if gold + predicted else 0.0


def macro(counts, labels):
    labels = list(labels)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("A nonempty, unique label vocabulary is required.")
    gold, predicted, correct = counts
    if (set(gold) | set(predicted)) - set(labels):
        raise ValueError("Found a label outside the configured vocabulary.")
    return sum(fscore(correct[k], gold[k], predicted[k]) for k in labels) / len(labels)


def counts():
    return Counter(), Counter(), Counter()


def classify(pairs, labels):
    gold, predicted, correct = counts()
    for g, p in pairs:
        gold[g] += 1
        predicted[p] += 1
        correct[g] += g == p
    return macro((gold, predicted, correct), labels)


def text_score(refs, preds):
    from bert_score import score
    if any(not isinstance(s, str) or not s.strip() for s in refs + preds):
        raise ValueError("Generated text and reference text must be nonempty strings.")
    return float(score(preds, refs, lang="en", batch_size=8, verbose=False)[2].mean())
