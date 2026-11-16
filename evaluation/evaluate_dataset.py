import argparse
import json
import pandas as pd
import spacy
from pathlib import Path

from validators.falc_checker import load_rulebook, check_text
from validators import falc_checker

from validators.falc_checker import SENTENCE_RULES

def evaluate_pair(row, nlp, book, rules):
    """Evaluate linguistic quality and formatting separately."""

    standard = check_text(
        str(row["paragraph_standard"]), nlp, book, rules
    )

    falc = check_text(
        str(row["paragraph_falc"]), nlp, book, rules
    )

    # Active linguistic rules: exclude document-level MP8
    linguistic_rules = list(SENTENCE_RULES.keys())

    total_weight = sum(
        rules[rid]["weight"] for rid in linguistic_rules
    )

    def linguistic_score(result):
        failed = set(result["rules_failed"])

        passed_weight = sum(
            rules[rid]["weight"]
            for rid in linguistic_rules
            if rid not in failed
        )

        return round(100 * passed_weight / total_weight, 1)

    standard_score = linguistic_score(standard)
    falc_score = linguistic_score(falc)

    # MP8 formatting compliance, separately
    standard_mp8_failed = "MP8" in standard["rules_failed"]
    falc_mp8_failed = "MP8" in falc["rules_failed"]

    return {
        "pair_uid": row["pair_uid"],
        "pair_id": row["pair_id"],
        "alignment_confidence": row["alignment_confidence"],

        "standard_score": standard_score,
        "falc_score": falc_score,
        "score_difference": round(falc_score - standard_score, 1),

        "standard_mp8_failed": standard_mp8_failed,
        "falc_mp8_failed": falc_mp8_failed,

        "standard_failed_rules": json.dumps(
            standard["rules_failed"], ensure_ascii=False
        ),
        "falc_failed_rules": json.dumps(
            falc["rules_failed"], ensure_ascii=False
        ),

        "standard_text": row["paragraph_standard"],
        "falc_text": row["paragraph_falc"],
    }

def print_summary(results, label):
    """Print aggregate comparison statistics."""

    if results.empty:
        print(f"\nNo results available for {label}.")
        return

    differences = results["score_difference"]

    print(f"\n{'=' * 55}")
    print(f"RESULTS: {label}")
    print(f"{'=' * 55}")

    print(f"Number of pairs: {len(results)}")

    print(
        f"Average standard score: "
        f"{results['standard_score'].mean():.2f}%"
    )

    print(
        f"Average FALC score: "
        f"{results['falc_score'].mean():.2f}%"
    )

    print(
        f"Average score difference: "
        f"{differences.mean():+.2f} percentage points"
    )

    print(
        f"FALC scored higher: "
        f"{(differences > 0).mean() * 100:.1f}%"
    )

    print(
        f"Equal scores: "
        f"{(differences == 0).mean() * 100:.1f}%"
    )

    print(
        f"FALC scored lower: "
        f"{(differences < 0).mean() * 100:.1f}%"
    )


def build_rule_summary(results, rule_ids):
    """Calculate document-level failure rates for each rule."""

    rows = []

    for rule_id in rule_ids:
        standard_failures = results[
            "standard_failed_rules"
        ].apply(lambda x: rule_id in json.loads(x))

        falc_failures = results[
            "falc_failed_rules"
        ].apply(lambda x: rule_id in json.loads(x))

        rows.append({
            "rule_id": rule_id,
            "standard_failure_rate": round(
                standard_failures.mean() * 100, 2
            ),
            "falc_failure_rate": round(
                falc_failures.mean() * 100, 2
            ),
            "difference_percentage_points": round(
                (falc_failures.mean() - standard_failures.mean()) * 100,
                2
            )
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent.parent / "data" / "falc_fr.csv")
    )

    parser.add_argument(
        "--model",
        default="fr_core_news_md"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.80
    )

    args = parser.parse_args()

    print("Loading dataset...")

    df = pd.read_csv(args.dataset)

    # Keep complete text pairs.
    df = df.dropna(
        subset=["paragraph_standard", "paragraph_falc"]
    ).copy()

    df["alignment_confidence"] = pd.to_numeric(
        df["alignment_confidence"],
        errors="coerce"
    )

    print(f"Pairs to evaluate: {len(df)}")

    print(f"Loading spaCy model: {args.model}")

    # Load the NLP model only once.
    nlp = spacy.load(args.model)

    book, rules = load_rulebook()

    print("Starting evaluation...")

    records = []

    for index, (_, row) in enumerate(df.iterrows(), start=1):

        records.append(
            evaluate_pair(row, nlp, book, rules)
        )

        if index % 50 == 0:
            print(f"Processed {index}/{len(df)} pairs")

    results = pd.DataFrame(records)

    # Save pair-level results.
    results.to_csv(
        "falc_comparison_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nPair-level results saved.")

    # Full dataset comparison.
    print_summary(results, "FULL DATASET")

    # High-confidence comparison.
    high_confidence = results[
        results["alignment_confidence"] >= args.confidence
    ].copy()

    print_summary(
        high_confidence,
        f"HIGH-CONFIDENCE PAIRS (>= {args.confidence})"
    )

    # Rule-level comparison.
    rule_ids = list(falc_checker.SENTENCE_RULES.keys()) + ["MP8"]

    rule_summary = build_rule_summary(results, rule_ids)

    rule_summary.to_csv(
        "falc_rule_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nRule-level results:")
    print(rule_summary.to_string(index=False))

    print("\nFiles generated:")
    print("- falc_comparison_results.csv")
    print("- falc_rule_comparison.csv")


if __name__ == "__main__":
    main()
