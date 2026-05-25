# RAG Shakespeare Project

## Overview
This project implements a Retrieval-Augmented Generation (RAG) based question-answering system using Shakespearean play dialogues. Semantic retrieval and evidence-grounded response generation techniques have been integrated to improve contextual understanding and reduce hallucination during language model response generation.

Selected Shakespeare plays:
- Hamlet
- Macbeth
- Romeo and Juliet

---

## Objectives
- Implement a basic RAG pipeline
- Generate semantic embeddings for dialogue retrieval
- Perform vector similarity search using FAISS
- Integrate FLAN-T5 for grounded response generation
- Evaluate contextual retrieval and response quality

---

## Technologies Used
- Python
- Google Colab
- Sentence Transformers
- all-MiniLM-L6-v2
- FAISS
- FLAN-T5
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## RAG Pipeline Architecture

### 1. Dataset Preparation
- Shakespeare dialogue dataset preprocessing
- Missing value handling
- Dialogue chunk generation
- Metadata structuring

### 2. Embedding Generation
- SentenceTransformer model:
  `all-MiniLM-L6-v2`
- 384-dimensional vector embeddings generated

### 3. Semantic Retrieval
- FAISS vector indexing
- Top-k similarity retrieval
- Evidence extraction using metadata

### 4. LLM Response Generation
- FLAN-T5 integration
- Prompt engineering
- Context-aware answer generation

---

## Dataset Information
The dataset contains:
- Play name
- Character name
- Act
- Scene
- Dialogue text
- Sentence metadata

Approximately:
- 9,487 dialogue chunks
- 384-dimensional embeddings

---

## Visualisations Included
- Dialogue Distribution by Play
- Top Frequent Speakers
- Dialogue Distribution Across Acts
- PCA-based Embedding Visualisation

## Setup Instructions
- git clone https://github.com/Ansari150102/RAG_Shakespeare_Project.git
- cd RAG_Shakespeare_Project
- pip install pandas numpy faiss-cpu sentence-transformers==2.7.0 transformers==4.41.2 torch matplotlib scikit-learn
- python rag_cli.py

