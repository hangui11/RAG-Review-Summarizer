# Review Summarizer using Retrieval-Augmented Generation (RAG)

![Framework Architecture](images/framework.png)

## 📌 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** framework designed to summarize large-scale e-commerce reviews based on specific user queries. 

Traditional summarization models (like TextRank or BART) often struggle with specific user questions because they either hallucinate answers or provide generic summaries of the entire text. This project solves that by:
1.  **Retrieving** only the most relevant reviews from a large corpus using vector embeddings.
2.  **Augmenting** a Generative LLM (Qwen) with this specific context.
3.  **Generating** a precise, fact-based summary.

We benchmark this RAG approach against two baselines: **TextRank** (Extractive) and **BART** (Abstractive) across the Amazon Review (Beauty and Music) and SAMSum datasets.

---

## 🚀 Features
* **RAG Pipeline:** Uses `sentence-transformers` for embeddings and `FAISS` for dense vector retrieval.
* **LLM Integration:** Implements **Qwen3-4B-Instruct** with 4-bit quantization for efficient local inference.
* **Benchmarking:** Compares RAG against:
    * **TextRank:** Graph-based extractive summarization.
    * **BART:** Transformer-based abstractive summarization.
* **Evaluation:**  **Quantitative:** ROUGE scores on the SAMSum dataset.
    * **Qualitative:** An "LLM-as-a-Judge" protocol to grade accuracy, faithfulness, and relevance on a scale of 1-10.

---

## 🛠️ Installation & Prerequisites

### 1. System Requirements
* **Python 3.8+**

### 2. Install Dependencies
Run the following commands to install the required libraries:

```bash
# Core Machine Learning & NLP Libraries
pip install torch datasets sentence_transformers transformers faiss numpy pandas

# Summarization & Evaluation Utilities
pip install spacy pytextrank evaluate rouge_score bert_score absl-py

# Quantization & Optimization
pip install bitsandbytes accelerate
```

### 3. Download Spacy Model
This is required for the TextRank baseline:

```bash
python -m spacy download en_core_web_sm
```
---
## 📂 Repository Structure
```bash
├── images/
│   ├── rag_framework.png           # Basic Architecture of RAG
│   ├── framework                   # Architecture diagram
├── RAG_Review_Summarization.ipynb  # Main Jupyter Notebook
└── README.md                       # This file
```
---
## 📖 Usage

1. **Launch the Notebook:**
Open `RAG_Review_Summarization.ipynb` in Jupyter Lab or VS Code.
2. **Run All Cells:**
The notebook is self-contained. It will:
    * **Download** the Amazon Reviews (Beauty & Music) and SAMSum datasets automatically.
    * **Build** the FAISS vector index (Offline Phase).
    * **Run** the "Before vs. After" comparison loop.
    * **Generate** the final results table comparing TextRank, BART, and Qwen (RAG).




---

## 📊 Methodology

The project follows a comparative analysis between **Naive Summarization** (No Search) and **RAG Summarization** (With Search):

| Component | Model / Technology | Description |
| --- | --- | --- |
| **Retriever** | `all-MiniLM-L6-v2` | Bi-encoder that maps queries and reviews to 384-dim vectors. |
| **Index** | `FAISS (IndexFlatL2)` | Performs exact Euclidean distance search to find top-k reviews. |
| **Generator** | `Qwen3-4B-Instruct-2507` | 3B parameter LLM optimized for instruction following. |
| **Baselines** | `TextRank`, `BART-Large` | Used for comparison to prove the efficacy of RAG. |

---

## 🏆 Key Results

The experiments demonstrate that **RAG significantly outperforms traditional methods** for query-focused tasks:

* **TextRank** fails to synthesize information, often returning disjointed sentences.
* **BART** tends to hallucinate or generate generic summaries when relevant context is missing.
* **RAG (Qwen)** consistently achieves high Judge Scores (9/10), accurately synthesizing conflicting user opinions (e.g., balancing "good sound" with "high price").

