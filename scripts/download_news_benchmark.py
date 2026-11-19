"""Download a version-pinned FinancialPhraseBank research benchmark; never execute dataset code."""
from pathlib import Path
import sys
import hashlib
import io
import json
import zipfile
import requests
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.artifacts import atomic_json


def main():
    repository = "takala/financial_phrasebank"
    response = requests.get(f"https://huggingface.co/api/datasets/{repository}", timeout=30)
    response.raise_for_status()
    revision = response.json()["sha"]
    url = f"https://huggingface.co/datasets/{repository}/resolve/{revision}/data/FinancialPhraseBank-v1.0.zip"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    # Read the known member only; do not extract arbitrary archive paths.
    member = next(name for name in archive.namelist() if name.endswith("Sentences_AllAgree.txt"))
    text = archive.read(member).decode("latin-1")
    examples = []
    for line in text.splitlines():
        if not line.strip():
            continue
        sentence, label = line.rsplit("@", 1)
        label = label.strip()
        if label not in {"negative", "neutral", "positive"}:
            raise ValueError(f"Unexpected label: {label}")
        examples.append({"text":sentence.strip(), "label":label})
    output = ROOT / "data" / "benchmarks"
    output.mkdir(parents=True, exist_ok=True)
    (output / "FinancialPhraseBank-v1.0.zip").write_bytes(response.content)
    attribution = {"dataset":"FinancialPhraseBank", "authors":"Pekka Malo, Ankur Sinha, Pekka Korhonen, Jyrki Wallenius, Pekka Takala",
                   "paper":"Good debt or bad debt: Detecting semantic orientations in economic texts (2014)",
                   "source":"https://huggingface.co/datasets/takala/financial_phrasebank", "revision":revision,
                   "license":"CC-BY-NC-SA-3.0", "license_url":"https://creativecommons.org/licenses/by-nc-sa/3.0/",
                   "subset":"sentences_allagree", "sample_count":len(examples),
                   "archive_sha256":hashlib.sha256(response.content).hexdigest(),
                   "scope":"Expert-labeled English financial sentences; not an Indian-market return-prediction benchmark."}
    atomic_json(output / "financial_phrasebank_provenance.json", attribution)
    atomic_json(output / "financial_phrasebank.json", {"kind":"expert_labeled_financial_phrasebank_allagree",
                "provenance":attribution, "sentiment":examples, "retrieval":[]})
    print(json.dumps(attribution, indent=2))


if __name__ == "__main__":
    main()
