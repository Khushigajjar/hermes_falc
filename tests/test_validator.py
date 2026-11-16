import spacy
from pprint import pprint
from validators.meaning_validator import validate_meaning

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

nlp = spacy.load("fr_core_news_md")

REVIEWER_MODEL = "phi3:mini"
OLLAMA_URL = "http://localhost:11434/api/generate"

for name, candidate in TESTS.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    result = validate_meaning(
        ORIGINAL,
        candidate,
        nlp,
        REVIEWER_MODEL,
        OLLAMA_URL
    )

    pprint(result, sort_dicts=False)
