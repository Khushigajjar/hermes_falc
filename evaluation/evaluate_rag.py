import spacy

from validators.falc_checker import load_rulebook, check_text, print_report

# The RAG-generated text
rag_text = """
La ville a décidé de créer un service pour aider les personnes âgées lors de leurs procédures administratives.

Ce service est personnalisé.

Ce service est pour les personnes âgées uniquement.
"""

print("Loading French NLP model...")
nlp = spacy.load("fr_core_news_md")

print("Loading FALC rulebook...")
book, rules = load_rulebook()

print("\nEvaluating RAG-generated text...")

report = check_text(rag_text, nlp, book, rules)

print_report(report, rules)

print("\n========== RAG EVALUATION SUMMARY ==========")
print(f"FALC score: {report['score_percent']}%")
print(f"Failed rules: {report['rules_failed']}")
print("============================================")
