import ollama
from rag.rag_retriever import create_vector_database, retrieve_examples

# 1. Build/load the semantic retrieval database
collection = create_vector_database()

original_text = (
    "La municipalité a décidé de mettre en place un dispositif "
    "permettant aux personnes âgées de bénéficier d'un accompagnement "
    "personnalisé dans leurs démarches administratives."
)

# 2. Retrieve relevant examples
examples = retrieve_examples(original_text, collection, top_k=3)

# 3. Format retrieved examples as context
context = ""

for i, example in enumerate(examples, 1):
    context += f"""
EXAMPLE {i}
Original text:
{example['standard']}

FALC version:
{example['falc']}
"""

# 4. Build the generation prompt
prompt = f"""
You are a French FALC simplification assistant.

Simplify the original text using the retrieved examples as guidance.

IMPORTANT RULES:
- Preserve all information from the original text.
- Do not invent facts or change the meaning.
- Use simple, common French vocabulary.
- Use short sentences, ideally no more than 15 words.
- Express one main idea per sentence.
- Use direct and clear language.
- The retrieved examples are guidance, not text to copy.
- Do not remove important information.

RETRIEVED EXAMPLES:
{context}

ORIGINAL TEXT:
{original_text}

FALC SIMPLIFICATION:
"""

# 5. Generate with Mistral
print("\nGenerating FALC text with RAG context...\n")

response = ollama.chat(
    model="mistral:latest",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("========== RAG GENERATION ==========")
print(response["message"]["content"])
print("====================================")
