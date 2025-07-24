import pandas as pd
import re

# Daftar stopword 
STOPWORDS = {
    "lan",      # dan
    "utawa",    # atau
    "nanging",  # tetapi
    "amarga",   # karena
    "dadi",     # jadi
    "nalika",   # ketika / saat
    "supaya",   # supaya
    "sanajan",  # meskipun
    "yen",      # jika
    "nek",      # kalau
    "kajaba",   # kecuali
    "bisan",    # bahkan
    "tur",      # lalu
    "salajengipun",  # selanjutnya
    "banjur",   # kemudian
    "bareng",   # bersamaan
    "kejaba",   # kecuali
    "mulaning", # sebab
    "awit",     # sejak / karena
    "amrih",    # agar / supaya
    "padahal",  # padahal
    "merga"     # karena
}

def preprocess_dataset(csv_path="data/dataset_jawa_cleaned.csv"):
    df = pd.read_csv(csv_path)

    contexts = []
    tokens_target = []
    labels = []

    for _, row in df.iterrows():
        # Lowercasing kalimat
        kalimat = row["kalimat"].lower()

        # Noise removal
        kalimat = re.sub(r"[^a-z\s]", " ", kalimat)

        # Menghapus spasi berlebih
        kalimat = re.sub(r"\s+", " ", kalimat).strip()

        # Tokenisasi
        tokens = kalimat.split()

        label = row["label"]

        for i in range(len(tokens)):
            token = tokens[i]

            # Stopword removal
            if token in STOPWORDS:
                continue

            if i == 0:
                context = " ".join(tokens[i:i+2])
            elif i == len(tokens) - 1:
                context = " ".join(tokens[i-1:i+1])
            else:
                context = " ".join(tokens[i-1:i+2])

            contexts.append(context)
            tokens_target.append(token)
            labels.append(label)

    return contexts, tokens_target, labels
