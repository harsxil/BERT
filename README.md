
\# Self-Attention: Understanding Context Inside BERT

A presentation-ready, technically accurate case study demonstrating **what self-attention does inside BERT**.

---

## Academic Thesis & Distinction

> **Self-Attention is a mechanism.**
> **BERT is a pretrained bidirectional Transformer encoder that stacks multi-head self-attention across 12 layers to produce contextual word representations.**

This demo does **not** treat Self-Attention and BERT as competing models. Instead, it demonstrates how BERT applies multi-head self-attention to incorporate surrounding context, allowing the **same surface word** to receive **different contextual representations** across different sentences.

---

## Core Demonstration

The case study analyzes two classic sentences containing the same target word:

- **Sentence A:** *"The animal didn't cross the road because it was tired."*
- **Sentence B:** *"The animal didn't cross the road because it was raining."*
- **Target Word:** `it`

### Narrative Flow:
1. **Tokenization:** Real WordPiece tokenization via `BertTokenizerFast` with precise character-to-token offset mapping.
2. **Contextual Representations:** Hidden representations extracted directly from BERT's final encoder layer (Layer 12, dimension 768).
3. **Cosine Similarity:** Real cosine similarity between $\text{BERT}(\text{"it"} \mid A)$ and $\text{BERT}(\text{"it"} \mid B)$ ($\approx 0.6137$), compared to a static embedding baseline ($1.0000$).
4. **Look Inside BERT (Attention Tensors):** True self-attention matrices extracted from `outputs.attentions[layer][head]`, with interactive Layer (0–11) and Head (0–11) selection.
5. **Attention Distributions:** Exact attention distribution row for `"it"` over all sequence tokens (verifying $\sum \alpha_i \approx 1.0$ and non-uniform standard deviation $> 0$).
6. **Theoretical Foundation:** The $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ formulation and how Query, Key, and Value projections allow tokens to mix information.

---

## Architecture & Technology Stack

- **Backend:** FastAPI, PyTorch, Hugging Face `transformers` (`bert-base-uncased`), NumPy.
- **Frontend:** React 18, Vite 6, modern responsive Dark-Mode CSS design system.
- **No external LLM APIs, no mock data, no synthetic attention scores.**

---

## Setup & Running

### 1. Backend Setup

Ensure you have Python 3.10+ (tested on Python 3.13):

```powershell
# From project root
cd backend

# Activate your virtual environment (e.g., TA)
..\..\TA\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
python main.py
```
The backend API will be available at `http://localhost:8000`.

### 2. Frontend Setup

```powershell
# In a separate terminal, from project root
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## Verification & Testing

### Backend Unit Tests
Run the automated test suite verifying tokenization, vector extraction, cosine similarity, and real BERT attention tensor properties:

```powershell
cd backend
..\..\TA\Scripts\python.exe test_backend.py
```

### Frontend Production Build
Validate that the React application compiles cleanly without warnings:

```powershell
cd frontend
npm run build
```
