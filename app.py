from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import re
import os

app = Flask(__name__)
model = joblib.load("model/naive_bayes_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Load daftar token unik dari dataset
def load_tokens(path="data/dataset_jawa_cleaned.csv"):
    df = pd.read_csv(path)
    tokens = set()
    for kalimat in df["kalimat"]:
        tokens.update(kalimat.strip().lower().split())
    return sorted(tokens)

# Token dataset untuk pengecekan "tidak terdeteksi"
token_list = load_tokens()

@app.route("/", methods=["GET", "POST"])
def index():
    hasil = {"ngoko": [], "madya": [], "kramainggil": []}
    per_token_result = []
    final_distribution = {}
    kalimat = ""

    if request.method == "POST":
        kalimat = request.form["kalimat"]
        tokens = kalimat.strip().lower().split()
        label_set = model.classes_
        counter = {label: 0 for label in label_set}

        for i in range(len(tokens)):
            token = tokens[i]
            # Cek apakah token ada di dataset
            if token not in token_list:
                per_token_result.append({
                    'token': token,
                    label_set[0]: 0.0,
                    label_set[1]: 0.0,
                    label_set[2]: 0.0,
                    'pred': "Tidak Terdeteksi"
                })
                continue

            # Buat context (bigram)
            context = (
                " ".join(tokens[i:i+2]) if i == 0 else
                " ".join(tokens[i-1:i+1]) if i == len(tokens)-1 else
                " ".join(tokens[i-1:i+2])
            )

            x_input = vectorizer.transform([context])

            if x_input.nnz > 0:
                probs = model.predict_proba(x_input)[0]
                pred = label_set[np.argmax(probs)]
                counter[pred] += 1

                per_token_result.append({
                    'token': token,
                    label_set[0]: round(probs[0]*100, 1),
                    label_set[1]: round(probs[1]*100, 1),
                    label_set[2]: round(probs[2]*100, 1),
                    'pred': pred
                })
            else:
                # Jika tidak ada fitur terdeteksi di TF-IDF
                per_token_result.append({
                    'token': token,
                    label_set[0]: 0.0,
                    label_set[1]: 0.0,
                    label_set[2]: 0.0,
                    'pred': "Tidak Terdeteksi"
                })

        total = sum(counter.values())
        if total > 0:
            final_label = max(counter, key=counter.get)
            confidence = round(counter[final_label] / total, 3)
            final_distribution = {
                label: round((count/total)*100, 1) for label, count in counter.items()
            }
            final_distribution['final_label'] = final_label
            final_distribution['confidence'] = confidence
        else:
            final_distribution = {
                label: 0.0 for label in label_set
            }
            final_distribution['final_label'] = "Tidak Terdeteksi"
            final_distribution['confidence'] = 0.0

    return render_template("index.html",
        kalimat=kalimat,
        hasil=hasil,
        per_token_result=per_token_result,
        final_distribution=final_distribution
    )

@app.route("/suggest")
def suggest():
    query = request.args.get("q", "").lower()
    cleaned_tokens = [re.sub(r"[^a-zA-Z]", "", token) for token in token_list]
    suggestions = [token for token in cleaned_tokens if token.startswith(query)]
    return jsonify(suggestions[:10])

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))