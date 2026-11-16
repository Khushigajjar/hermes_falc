import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd
import spacy

from validators.falc_checker import check_text, load_rulebook


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "data" / "falc_fr.csv"
OUTPUT = Path("model_comparison_results.csv")

MODELS = ["mistral:latest", "phi3:mini"]

SAMPLE_SIZE = 50
CONFIDENCE_THRESHOLD = 0.80
RANDOM_SEED = 42

#
RUN_LIMIT = None

OLLAMA_URL = "http://localhost:11434/api/generate"


def simplify_text(text, model):
    prompt = f"""
Tu es un assistant spécialisé dans la simplification FALC.

Réécris le texte en français facile à comprendre.

Règles obligatoires :
- Chaque phrase doit contenir au maximum 15 mots.
- Utilise des mots courants et faciles à comprendre.
- Utilise la voix active.
- Présente une seule idée par phrase.
- Écris chaque phrase sur une nouvelle ligne.
- Conserve le temps des verbes du texte original.
- Conserve toutes les informations importantes.
- Explique les mots difficiles avec des mots simples.
- Ne change pas le sens du texte.
- N'ajoute aucune information.
- Vérifie les espaces entre tous les mots.
- Réponds uniquement avec le texte simplifié.

Texte :
{text}
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"].strip()


def count_words(text):
    return len(re.findall(r"\b[\wÀ-ÿ'-]+\b", text))


def evaluate_text(text, nlp, book, rules):
    report = check_text(text, nlp, book, rules)
    doc = nlp(text)

    return {
        "checker_score": report["score_percent"],
        "rules_failed": json.dumps(
            report["rules_failed"], ensure_ascii=False
        ),
        "word_count": count_words(text),
        "sentence_count": len(list(doc.sents))
    }


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATASET, encoding="utf-8")

    df["alignment_confidence"] = pd.to_numeric(
        df["alignment_confidence"], errors="coerce"
    )

    eligible = df[
        df["alignment_confidence"] >= CONFIDENCE_THRESHOLD
    ].copy()

    selected = eligible.sample(
        n=min(SAMPLE_SIZE, len(eligible)),
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    if RUN_LIMIT is not None:
        selected = selected.head(RUN_LIMIT)

    print(f"Eligible pairs: {len(eligible)}")
    print(f"Pairs selected for this run: {len(selected)}")

    print("\nLoading French NLP model...")
    nlp = spacy.load("fr_core_news_md")

    print("Loading FALC rulebook...")
    book, rules = load_rulebook()

    # Load previous results so interrupted experiments can resume.
    if OUTPUT.exists():
        results = pd.read_csv(OUTPUT, encoding="utf-8-sig").to_dict(
            orient="records"
        )
    else:
        results = []

    completed = {
        (r["pair_uid"], r["model"])
        for r in results
        if r.get("status") == "success"
    }

    for index, row in selected.iterrows():
        pair_uid = row["pair_uid"]
        original = str(row["paragraph_standard"])
        reference = str(row["paragraph_falc"])

        print(f"\nPair {index + 1}/{len(selected)}: {pair_uid}")

        # Evaluate original and reference texts once.
        original_metrics = evaluate_text(original, nlp, book, rules)
        reference_metrics = evaluate_text(reference, nlp, book, rules)

        for model in MODELS:
            if (pair_uid, model) in completed:
                print(f"  Skipping {model}: already completed")
                continue

            print(f"  Generating with {model}...")

            start = time.time()

            try:
                simplified = simplify_text(original, model)
                elapsed = round(time.time() - start, 2)

                metrics = evaluate_text(
                    simplified, nlp, book, rules
                )

                result = {
                    "pair_uid": pair_uid,
                    "alignment_confidence": row[
                        "alignment_confidence"
                    ],
                    "model": model,
                    "status": "success",
                    "generation_time_seconds": elapsed,
                    "original_text": original,
                    "reference_falc_text": reference,
                    "generated_text": simplified,
                    "original_checker_score":
                        original_metrics["checker_score"],
                    "reference_checker_score":
                        reference_metrics["checker_score"],
                    **metrics
                }

                print(
                    f"  Score: {metrics['checker_score']}% | "
                    f"Time: {elapsed}s"
                )

            except Exception as error:
                result = {
                    "pair_uid": pair_uid,
                    "alignment_confidence": row[
                        "alignment_confidence"
                    ],
                    "model": model,
                    "status": "error",
                    "generation_time_seconds": round(
                        time.time() - start, 2
                    ),
                    "original_text": original,
                    "reference_falc_text": reference,
                    "generated_text": "",
                    "error": str(error)
                }

                print(f"  ERROR: {error}")

            results.append(result)

            # Save after every model to avoid losing progress.
            pd.DataFrame(results).to_csv(
                OUTPUT,
                index=False,
                encoding="utf-8-sig"
            )

    print("\n" + "=" * 55)
    print("EXPERIMENT RUN FINISHED")
    print("=" * 55)
    print(f"Results saved to: {OUTPUT}")

    successful = [
        r for r in results if r.get("status") == "success"
    ]

    if successful:
        summary = pd.DataFrame(successful)

        print("\nSuccessful generations by model:")
        print(
            summary.groupby("model").agg(
                generations=("pair_uid", "count"),
                average_score=("checker_score", "mean"),
                average_time=("generation_time_seconds", "mean")
            ).round(2)
        )


if __name__ == "__main__":
    main()
