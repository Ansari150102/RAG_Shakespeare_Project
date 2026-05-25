import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

# LOAD DATASET

print("Loading Dataset...")

df = pd.read_csv("Dataset/shakespeare_plays.csv")

plays = ["Hamlet", "Macbeth", "Romeo and Juliet"]

df = df[df["play_name"].isin(plays)]

df = df.dropna(subset=["text", "character"])
df = df.reset_index(drop=True)

# CREATE CHUNKS

chunks = []

for _, row in df.iterrows():

    chunk = {
        "play": row["play_name"],
        "act": row["act"],
        "scene": row["scene"],
        "speaker": row["character"],
        "text": row["text"]
    }

    chunks.append(chunk)

texts = [chunk["text"] for chunk in chunks]

print(f"Total Chunks Loaded: {len(texts)}")

# EMBEDDING MODEL

print("Loading Embedding Model...")

embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)

# FAISS INDEX

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("FAISS Index Ready")

# LOAD FLAN-T5

print("Loading FLAN-T5 Model...")

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer
)

print("RAG System Ready")

# RETRIEVAL FUNCTION

def retrieve(query, k=3):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        k
    )

    results = []

    for idx in indices[0]:
        results.append(chunks[idx])

    return results

# QUESTION ANSWERING

def ask_question(query):

    retrieved_docs = retrieve(query)

    context = ""

    for doc in retrieved_docs:

        context += f"""
Play: {doc['play']}
Act: {doc['act']}
Scene: {doc['scene']}
Speaker: {doc['speaker']}

Text:
{doc['text']}
"""

    print("\n" + "=" * 60)
    print("RETRIEVED EVIDENCE")
    print("=" * 60)

    print(context)

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    response = generator(
        prompt,
        max_length=100
    )

    answer = response[0]['generated_text']

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)

    print(answer)

# CLI LOOP

print("\nType 'exit' to quit.\n")

while True:

    query = input("Ask a Shakespeare Question: ")

    if query.lower() == "exit":
        break

    ask_question(query)