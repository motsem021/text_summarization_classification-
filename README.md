📰 NLP Topic Classifier & Summarizer

A Python application that allows users to predict topics of sentences and summarize articles using Natural Language Processing (NLP) techniques.
It uses TF-IDF, Non-negative Matrix Factorization (NMF), and extractive summarization.

🚀 Features

Topic Classification: Predict the topic of a given sentence based on trained NMF model.

Article Summarization: Generate a concise summary of articles from the NPR dataset.

GUI Interface: User-friendly Tkinter interface for input and output.

📂 Project Structure
.
├── text_classification_summarizer.py  # Main Python script (your code)
├── npr.csv                            # Dataset with articles
├── README.md                          # Project documentation

🛠 Installation

Clone the repository:

git clone https://github.com/motsem021/text_summarization_classification-.git
cd text_summarization_classification-


Install dependencies:

pip install pandas scikit-learn nltk tk


Download NLTK data:

import nltk
nltk.download('punkt')
nltk.download('stopwords')


Make sure npr.csv is in the same folder as the script.

▶️ Usage

Run the application:

python text_classification_summarizer.py

Features in GUI

Predict Sentence Topic

Enter a sentence in the text box.

Click Predict Topic.

The predicted topic label will appear below.

Summarize Article

Enter an article index (integer) in the box.

Click Summarize Article.

The generated summary will appear below.

⚙️ How It Works

Topic Modeling

TF-IDF converts text into numerical features.

NMF finds latent topics in the articles.

Each article is assigned the topic with the highest probability.

Summarization

Extractive approach: scores sentences based on word frequency.

Returns the top 7 most relevant sentences as a summary.

📝 Dataset

Uses npr.csv containing news articles.

Expected column: Article.

🛠 Dependencies

Python 3.8+

pandas

scikit-learn

nltk

Tkinter (for GUI)

📄 License

This project is open-source under the MIT License.
