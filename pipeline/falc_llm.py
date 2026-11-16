import json
import urllib.request
import urllib.error

import spacy

from validators.falc_checker import check_text, load_rulebook


MODEL = "mistral:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"

TEXT = """
La municipalité a décidé de mettre en place
un dispositif permettant aux personnes âgées
de bénéficier d'un accompagnement personnalisé
dans leurs démarches administratives.
"""


def simplify_text(text):
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
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"].strip()


def main():
    print("Loading French NLP model...")
    nlp = spacy.load("fr_core_news_md")

    print("\nLoading FALC rulebook...")
    book, rules = load_rulebook()

    print("\nGenerating simplification with Ollama...")
    simplified = simplify_text(TEXT)

    print("\n" + "=" * 60)
    print("ORIGINAL TEXT")
    print("=" * 60)
    print(TEXT.strip())

    print("\n" + "=" * 60)
    print("LLM SIMPLIFIED TEXT")
    print("=" * 60)
    print(simplified)

    print("\n" + "=" * 60)
    print("FALC CHECKER RESULTS")
    print("=" * 60)

    report = check_text(simplified, nlp, book, rules)

    print(f"Score: {report['score_percent']}%")
    print(f"Rules checked: {report['rules_checked']}")
    print(f"Failed rules: {report['rules_failed']}")

    print("\nFailures by sentence:")
    for sentence in report["sentences"]:
        if sentence["failures"]:
            print(f"\n{sentence['sentence_id']}: {sentence['text']}")
            for failure in sentence["failures"]:
                print(f"  - {failure['evidence']}")

    print("\nNote:", report["note"])


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as error:
        print("\nCould not connect to Ollama.")
        print("Make sure Ollama is running.")
        print("Details:", error)
