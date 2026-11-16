import spacy

nlp = spacy.load("fr_core_news_md")

from validators.meaning_validator import check_protected_elements

ORIGINAL = (
    "La municipalité a décidé de mettre en place un dispositif "
    "permettant aux personnes âgées de bénéficier d'un "
    "accompagnement personnalisé dans leurs démarches administratives."
)

TESTS = {
    "Test 1 - Invented date": (
        "La municipalité a décidé de mettre en place ce dispositif "
        "le 15 mars 2026 pour aider les personnes âgées."
    ),
    "Test 2 - Missing information": (
        "La municipalité a décidé de mettre en place un dispositif "
        "pour aider les personnes âgées."
    ),
    "Test 3 - Changed meaning": (
        "Les personnes âgées ont décidé de mettre en place un "
        "dispositif pour aider la municipalité."
    ),
    "Test 4 - Repetition and commentary": (
        "La mairie a créé un dispositif pour aider les personnes âgées. "
        "La mairie a créé un dispositif pour aider les personnes âgées. "
        "Note: This rewrite may need further adjustments."
    ),
}

GOOD_REWRITE = (
    "La mairie a décidé de mettre en place un dispositif "
    "pour offrir une assistance personnalisée aux personnes âgées "
    "dans leurs procédures administratives."
)

result = check_protected_elements(
    ORIGINAL, GOOD_REWRITE, nlp
)

print("\n" + "=" * 55)
print("Test 5 - Valid Mistral rewrite")
print("Passed:", result["passed"])
print("Warnings:", result["warnings"])

nlp = spacy.load("fr_core_news_md")

for name, candidate in TESTS.items():
    result = check_protected_elements(
        ORIGINAL, candidate, nlp
    )

    print("\n" + "=" * 55)
    print(name)
    print("Passed:", result["passed"])

    print("Warnings:")
    for warning in result["warnings"]:
        print("-", warning)

    print(
        "Unsupported values:",
        result["potentially_unsupported_values"]
    )

    print(
        "Missing content:",
        result["content_anchor_check"][
            "potentially_missing_content_anchors"
        ]
    )

    print(
        "Duplicate sentences:",
        result["repetition_check"]["duplicate_sentences"]
    )

    print(
        "Meta-commentary detected:",
        bool(result["potential_meta_commentary_patterns"])
    )

    assert not result["passed"], f"{name} was not flagged!"

print("\nAll four flawed cases were flagged.")
