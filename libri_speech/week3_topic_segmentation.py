import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nltk.download("punkt")

TRANSCRIPTS_DIR = "data/transcripts"
SEGMENTS_DIR = "data/segments"
KEYWORDS_DIR = "data/keywords"
SUMMARIES_DIR = "data/summaries"

os.makedirs(SEGMENTS_DIR, exist_ok=True)
os.makedirs(KEYWORDS_DIR, exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)

# STEP 1: Combine all transcripts
full_text = ""

for file in os.listdir(TRANSCRIPTS_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(TRANSCRIPTS_DIR, file), encoding="utf-8") as f:
            for line in f:
                # Remove timestamps
                if "]" in line:
                    full_text += line.split("]")[1].strip() + " "

print(" Transcripts combined")

# STEP 2: Sentence splitting
sentences = nltk.sent_tokenize(full_text)
print(f" Total sentences: {len(sentences)}")

# STEP 3: Algorithm 1 – TF-IDF similarity (Baseline)
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

tfidf_similarities = []
for i in range(len(sentences) - 1):
    sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i + 1])[0][0]
    tfidf_similarities.append(sim)

tfidf_threshold = np.mean(tfidf_similarities)
tfidf_boundaries = [i + 1 for i, sim in enumerate(tfidf_similarities) if sim < tfidf_threshold]

print(" Algorithm 1 (TF-IDF) completed")

# STEP 4: Algorithm 2 – Embedding-based segmentation
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

embed_similarities = []
for i in range(len(sentences) - 1):
    sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
    embed_similarities.append(sim)

embed_threshold = np.mean(embed_similarities)
embed_boundaries = [i + 1 for i, sim in enumerate(embed_similarities) if sim < embed_threshold]

print("Algorithm 2 (Embedding-based) completed")

# STEP 5: Finalize segments (using embedding-based)
segments = []
start = 0

for boundary in embed_boundaries:
    segment_text = " ".join(sentences[start:boundary])
    if len(segment_text.strip()) > 0:
        segments.append(segment_text)
    start = boundary

segments.append(" ".join(sentences[start:]))

print(f" Total topic segments: {len(segments)}")

# STEP 6: Save segments
for i, seg in enumerate(segments):
    with open(f"{SEGMENTS_DIR}/segment_{i}.txt", "w", encoding="utf-8") as f:
        f.write(seg)

# STEP 7: Keyword extraction (TF-IDF)
for i, seg in enumerate(segments):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5)
    vectorizer.fit([seg])
    keywords = vectorizer.get_feature_names_out()

    with open(f"{KEYWORDS_DIR}/segment_{i}.txt", "w", encoding="utf-8") as f:
        f.write(", ".join(keywords))

# STEP 8: Summaries (first 2 sentences)
for i, seg in enumerate(segments):
    summary = " ".join(nltk.sent_tokenize(seg)[:2])
    with open(f"{SUMMARIES_DIR}/segment_{i}.txt", "w", encoding="utf-8") as f:
        f.write(summary)

print(" Week 3 completed: Topic segmentation, keywords, and summaries generated")
