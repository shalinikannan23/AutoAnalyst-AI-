import os
import shutil
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ---------- CONFIG ----------
genai.configure(api_key=xxx)
llm = genai.GenerativeModel("gemini-2.5-flash")

MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

VECTOR_DIR = "vector_db"
VECTOR_PATH = os.path.join(VECTOR_DIR, "faiss.index")
TEXT_PATH = os.path.join(VECTOR_DIR, "texts.npy")
CHAT_PATH = os.path.join(VECTOR_DIR, "chat.npy")


# ---------------- RESET ----------------
def reset_vector_store():
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
    os.makedirs(VECTOR_DIR, exist_ok=True)


# ---------------- BUILD DATA KNOWLEDGE ----------------
def build_dataset_knowledge(df, profile, eda, task_type):
    texts = []

    # Dataset overview
    texts.append(f"The dataset contains {profile['rows']} rows and {profile['columns']} columns.")
    texts.append(f"Target column is {profile['target_column']}.")
    texts.append(f"This is a {task_type} problem.")

    # Column level knowledge
    for col in df.columns:
        col_type = str(df[col].dtype)
        missing = int(df[col].isna().sum())
        unique = df[col].nunique()

        desc = f"Column {col} has data type {col_type}, {unique} unique values and {missing} missing values."

        if pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            minv = df[col].min()
            maxv = df[col].max()
            desc += f" Numeric stats: mean={round(mean,2)}, std={round(std,2)}, min={round(minv,2)}, max={round(maxv,2)}."

        else:
            top_vals = df[col].value_counts().head(3).to_dict()
            desc += f" Most frequent values: {top_vals}."

        texts.append(desc)

    # EDA knowledge
    texts.append(f"Outlier report: {eda['outliers']}.")
    texts.append(f"Correlation matrix: {eda['correlation'].to_dict()}.")

    # Sample-driven grounding
    sample_rows = df.sample(min(5, len(df))).to_dict(orient="records")
    texts.append(f"Sample records from dataset: {sample_rows}.")

    # Capability statements
    texts.append("This dataset is suitable for supervised machine learning modeling.")
    texts.append("This dataset can support predictive analytics and explainable AI.")
    texts.append("Potential tasks include classification, regression, anomaly detection and recommendation.")
    texts.append("Potential innovations include AutoML, feature explainability, bias detection and API deployment.")

    return texts


# ---------------- VECTOR STORE ----------------
def build_vector_store(texts):
    if os.path.exists(VECTOR_PATH):
        os.remove(VECTOR_PATH)
    if os.path.exists(TEXT_PATH):
        os.remove(TEXT_PATH)
    if os.path.exists(CHAT_PATH):
        os.remove(CHAT_PATH)

    embeddings = embedder.encode(texts)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)
    faiss.write_index(index, VECTOR_PATH)
    np.save(TEXT_PATH, np.array(texts, dtype=object))
    np.save(CHAT_PATH, np.array([], dtype=object))


def load_vector_store():
    if not os.path.exists(VECTOR_PATH):
        return None, None

    index = faiss.read_index(VECTOR_PATH)
    texts = np.load(TEXT_PATH, allow_pickle=True)
    return index, texts


# ---------------- CHAT MEMORY ----------------
def load_chat_history():
    if not os.path.exists(CHAT_PATH):
        return []
    return list(np.load(CHAT_PATH, allow_pickle=True))


def save_chat_history(history):
    np.save(CHAT_PATH, np.array(history, dtype=object))


# ---------------- RETRIEVE CONTEXT ----------------
def retrieve_context(query, k=6):
    index, texts = load_vector_store()
    if index is None or len(texts) == 0:
        return "No dataset knowledge available."

    q_emb = embedder.encode([query])
    _, indices = index.search(q_emb, k)

    return "\n".join([texts[i] for i in indices[0]])


# ---------------- LLM RESPONSE ----------------
def ask_llm(question):
    context = retrieve_context(question)
    history = load_chat_history()

    system_prompt = f"""
You are a senior data scientist assistant.
You MUST answer only using the dataset context below.
You must give:
- project ideas
- innovation ideas
- modeling suggestions
- data risks
- feature insights

DATASET CONTEXT:
{context}
"""

    chat_text = system_prompt + "\n\n"
    for h in history[-6:]:
        chat_text += f"{h['role'].upper()}: {h['content']}\n"

    chat_text += f"USER: {question}\nASSISTANT:"

    response = llm.generate_content(chat_text)
    answer = response.text

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    save_chat_history(history)

    return answer
