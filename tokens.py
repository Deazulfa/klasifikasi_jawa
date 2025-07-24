import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model dan vectorizer
with open("model/naive_bayes_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load token target dari CSV
tokens_target_df = pd.read_csv("data/tokens_target.csv")
tokens_target_set = set(tokens_target_df['token'].astype(str).str.lower().str.strip())

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    kalimat = data.get('kalimat', '').lower()
    tokens = kalimat.strip().split()

    results = []
    label_set = model.classes_
    label_counter = {label: 0 for label in label_set}

    for token in tokens:
        if token not in tokens_target_set:
            results.append({"token": token, "status": "unknown"})
            continue
        
        vec = vectorizer.transform([token])
        if vec.nnz == 0:
            results.append({"token": token, "status": "unknown"})
            continue

        probs = model.predict_proba(vec)[0]
        pred_label = label_set[np.argmax(probs)]
        label_counter[pred_label] += 1

        results.append({
            "token": token,
            "status": "known",
            "pred_label": pred_label,
            "probs": {label: float(probs[i]) for i, label in enumerate(label_set)}
        })

    total_known = sum(label_counter.values())
    label_summary = {label: round((label_counter[label]/total_known)*100, 2) if total_known > 0 else 0.0 for label in label_set}
    final_label = max(label_counter, key=label_counter.get) if total_known > 0 else "Tidak dapat diprediksi"
    confidence = round(label_counter[final_label]/total_known, 4) if total_known > 0 else 0.0

    return jsonify({
        "analysis_results": results,
        "label_summary": label_summary,
        "final_label": final_label,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
