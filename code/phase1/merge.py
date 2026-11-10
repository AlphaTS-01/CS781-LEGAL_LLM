"""Assemble the Phase 1 v2 predictions from the two LEXWeave generation passes."""
import argparse
import json
import re
import zipfile
from pathlib import Path

INVALID = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]')

def read_records(path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            return json.loads(archive.read("prediction.json"))
    return json.loads(path.read_text(encoding="utf-8"))

def text(record):
    return record["text"] if isinstance(record, dict) else record

def merge(primary, fallback, prompts):
    output = {}
    for case_id, prompt in prompts.items():
        first = text(primary.get(case_id, "")) if 1 <= int(case_id) <= 249 else ""
        second = text(fallback[case_id])
        a = INVALID.sub(" ", first).split()
        b = INVALID.sub(" ", second).split()
        if len(a) == 128 and not INVALID.search(first):
            words = a
        elif len(b) == 128 and not INVALID.search(second):
            words = b
        elif len(a) >= 128:
            words = a[:128]
        elif len(b) >= 128:
            words = b[:128]
        else:
            words = (a + b)[:128]
            if len(words) < 128:
                words += INVALID.sub(" ", text(prompt)).split()[-(128-len(words)):]
            while len(words) < 128:
                words += (words or ["The"])[:128-len(words)]
        output[case_id] = {"text": " ".join(words)}
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("primary", "fallback", "test", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    first, second, prompts = map(read_records, (args.primary, args.fallback, args.test))
    required = {k for k in prompts if 1 <= int(k) <= 249}
    if not required.issubset(first) or not set(prompts).issubset(second):
        parser.error("Missing primary or fallback predictions for the v2 selection.")
    predictions = merge(first, second, prompts)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("prediction.json", json.dumps(predictions, ensure_ascii=False))
