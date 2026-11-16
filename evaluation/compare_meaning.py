import json

from validators.meaning_validator import review_semantic_preservation

original_text = (
    "La municipalité a décidé de mettre en place un dispositif "
    "permettant aux personnes âgées de bénéficier d'un accompagnement "
    "personnalisé dans leurs démarches administratives."
)

# Output generated without RAG
no_rag_text = """
La ville décide d'offrir un accompagnement personnel aux personnes âgées
pour leurs procedures administratives.
"""

# Output generated with RAG
rag_text = """
La ville a décidé de créer un service pour aider les personnes âgées
lors de leurs procédures administratives.

Ce service est personnalisé.

Ce service est pour les personnes âgées uniquement.
"""

model = "phi3:mini"

experiments = {
    "WITHOUT RAG": no_rag_text,
    "WITH RAG": rag_text
}

for name, simplified_text in experiments.items():

    print(f"\n{'=' * 15} {name} {'=' * 15}")

    result = review_semantic_preservation(
        original_text=original_text,
        simplified_text=simplified_text,
        model=model
    )

    print(json.dumps(result, indent=4, ensure_ascii=False))
