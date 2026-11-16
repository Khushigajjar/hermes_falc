import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer


# Configuration
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "falc_fr.csv"
CHROMA_PATH = PROJECT_ROOT / "chroma_falc_db"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 3
MIN_CONFIDENCE = 0.80

COLLECTION_NAME = "falc_examples"


# Load the multilingual embedding model
print("Loading multilingual embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)


def load_knowledge_base():

    df = pd.read_csv(DATASET_PATH)

    # Keep eligible aligned examples.
    df = df[
        (df["validated_manually"] == True)
        & (df["hitl_status"] == "validated")
        & (df["license_compatible"] == True)
        & (df["alignment_confidence"] >= MIN_CONFIDENCE)
    ].copy()

    df = df.dropna(
        subset=["paragraph_standard", "paragraph_falc"]
    )

    df = df[
        (df["paragraph_standard"].str.strip() != "")
        & (df["paragraph_falc"].str.strip() != "")
    ]

    if df.empty:
        raise ValueError("No eligible examples found.")

    return df.reset_index(drop=True)


def create_vector_database():

    df = load_knowledge_base()

    print(f"Eligible examples: {len(df)}")

    # Create a persistent local ChromaDB database.
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    standard_texts = df["paragraph_standard"].tolist()

    print("Generating semantic embeddings...")

    embeddings = model.encode(
        standard_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Store standard paragraphs and aligned FALC versions.
    ids = df["pair_uid"].astype(str).tolist()

    documents = standard_texts

    metadatas = []

    for _, row in df.iterrows():

        metadatas.append({
            "pair_uid": str(row["pair_uid"]),
            "paragraph_falc": str(row["paragraph_falc"]),
            "alignment_confidence": float(
                row["alignment_confidence"]
            )
        })

    # Upsert prevents duplicate records on repeated runs.
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(f"Vector database ready: {collection.count()} examples")

    return collection


def retrieve_examples(query, collection, top_k=TOP_K):

    # Embed the input query.
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []

    if not results["ids"] or not results["ids"][0]:
        return retrieved

    for i in range(len(results["ids"][0])):

        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # For cosine distance, similarity = 1 - distance.
        similarity = 1.0 - distance

        retrieved.append({
            "similarity": float(similarity),
            "standard": results["documents"][0][i],
            "falc": metadata["paragraph_falc"],
            "confidence": metadata["alignment_confidence"],
            "pair_uid": metadata["pair_uid"]
        })

    return retrieved


if __name__ == "__main__":

    print("\nBuilding semantic retrieval database...")

    collection = create_vector_database()

    query = input(
        "\nEnter the original French text:\n"
    )

    examples = retrieve_examples(query, collection)

    print("\n========== SEMANTIC RAG RETRIEVAL ==========")

    print(f"Retrieved examples: {len(examples)}")

    for i, example in enumerate(examples, 1):

        print(f"\n--- Example {i} ---")

        print(
            f"Cosine similarity: "
            f"{example['similarity']:.4f}"
        )

        print(
            f"Alignment confidence: "
            f"{example['confidence']:.2f}"
        )

        print("\nSTANDARD:")
        print(example["standard"])

        print("\nFALC:")
        print(example["falc"])

    print("\n============================================")
