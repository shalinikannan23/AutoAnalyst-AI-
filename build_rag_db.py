from core.rag_engine import build_vector_store

texts = [
    "Titanic dataset classification project",
    "Customer churn prediction project",
    "House price regression project",
    "Fraud detection using machine learning",
    "Heart disease prediction dataset"
]

build_vector_store(texts)
print("Vector DB built")
