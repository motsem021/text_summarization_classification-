import pandas as pd
import tkinter as tk
from tkinter import messagebox, scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import heapq
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK data path
nltk.data.path.append("C:/Users/sesra/nltk_data")  # Adjust this to your path

# Load dataset
df = pd.read_csv("npr.csv", low_memory=False)

# TF-IDF and NMF
def tfidfvectorizer():
    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = tfidf.fit_transform(df['Article'])
    return dtm, tfidf

def run_nmf(tfidf, dtm):
    nmf_model = NMF(n_components=7, random_state=42)
    nmf_model.fit(dtm)
    topic_results = nmf_model.transform(dtm)
    df['Topic'] = topic_results.argmax(axis=1)
    topic_dict = {
        0: 'health',
        1: 'election',
        2: 'legislation',
        3: 'policy',
        4: 'candidates',
        5: 'music',
        6: 'education'
    }
    df['Topic Label'] = df['Topic'].map(topic_dict)
    return nmf_model, topic_dict

# Summarization
def summarization(index):
    if index >= len(df):
        return "Invalid index. Article not found."
    
    article_text = df['Article'].iloc[index]
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
    sentence_list = sent_tokenize(article_text)
    stop_words = set(stopwords.words('english'))

    word_frequencies = {}
    for word in word_tokenize(formatted_article_text.lower()):
        if word not in stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    max_freq = max(word_frequencies.values(), default=1)
    for word in word_frequencies:
        word_frequencies[word] /= max_freq

    sentence_scores = {}
    for sent in sentence_list:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies and len(sent.split(' ')) < 30:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

# Predict Topic
def predict_sentence_topic(sentence, tfidf, nmf_model, topic_dict):
    sentence_dtm = tfidf.transform([sentence])
    topic_prob = nmf_model.transform(sentence_dtm)
    topic_index = topic_prob.argmax()
    return topic_dict.get(topic_index, "Unknown")

# Prepare models
dtm, tfidf = tfidfvectorizer()
nmf_model, topic_dict = run_nmf(tfidf, dtm)

# GUI Setup
root = tk.Tk()
root.title("NLP Topic Classifier & Summarizer")

# --- Sentence Prediction ---
tk.Label(root, text="Enter a sentence to predict topic:").pack()
sentence_entry = tk.Entry(root, width=100)
sentence_entry.pack()

def on_predict():
    sentence = sentence_entry.get()
    if not sentence.strip():
        messagebox.showwarning("Input Error", "Please enter a sentence.")
        return
    label = predict_sentence_topic(sentence, tfidf, nmf_model, topic_dict)
    prediction_output.config(state='normal')
    prediction_output.delete('1.0', tk.END)
    prediction_output.insert(tk.END, f"Predicted Topic: {label}")
    prediction_output.config(state='disabled')

tk.Button(root, text="Predict Topic", command=on_predict).pack(pady=5)
prediction_output = scrolledtext.ScrolledText(root, height=3, width=100, state='disabled')
prediction_output.pack()

# --- Article Summarization ---
tk.Label(root, text="Enter article index to summarize (e.g., 0):").pack()
index_entry = tk.Entry(root, width=10)
index_entry.pack()

def on_summarize():
    try:
        index = int(index_entry.get())
        summary = summarization(index)
        summary_output.config(state='normal')
        summary_output.delete('1.0', tk.END)
        summary_output.insert(tk.END, summary)
        summary_output.config(state='disabled')
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid integer index.")

tk.Button(root, text="Summarize Article", command=on_summarize).pack(pady=5)
summary_output = scrolledtext.ScrolledText(root, height=10, width=100, state='disabled')
summary_output.pack()

root.mainloop()