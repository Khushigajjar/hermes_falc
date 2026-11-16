import ollama
import spacy

from validators.falc_checker import load_rulebook, check_text, print_report

original_text = (
    "La municipalité a décidé de mettre en place un dispositif "
    "permettant aux personnes âgées de bénéficier d'un accompagnement "
    "personnalisé dans leurs démarches administratives."
)

prompt = """
You are a French FALC simplification assistant.

Simplify the original text.

IMPORTANT RULES:
- Preserve all information from the original text.
- Do not invent facts or change the meaning.
- Use simple, common French vocabulary.
- Use short sentences, ideally no more than 15 words.
- Express one main idea per sentence.
- Use direct and clear language.

ORIGINAL TEXT:
""" + original_text + """

FALC SIMPLIFICATION:
"""

print("Generating text without RAG...")

response = ollama.chat(
    model="mistral:latest",
    messages=[{"role": "user", "content": prompt}]
)

generated_text = response["message"]["content"]

print("\n========== NO-RAG GENERATION ==========")
print(generated_text)

print("\nLoading checker...")
nlp = spacy.load("fr_core_news_md")
book, rules = load_rulebook()

report = check_text(generated_text, nlp, book, rules)

print("\n========== NO-RAG EVALUATION ==========")
print_report(report, rules)

print("\nNO-RAG SCORE:", report["score_percent"], "%")
print("FAILED RULES:", report["rules_failed"])
