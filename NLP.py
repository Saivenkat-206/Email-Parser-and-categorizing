# Over here we use the LDA/NMF model for assigning labels to the data that we got here.
# Unsupervised cuz we don't got a choice
# Non Negative Matrix Factorization 
# Email-Term Matrix ≈ Email-Topic Matrix x Topic-Term Matrix
# Each mail will be broken down into topics and those are broken into keywords

import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.decomposition import NMF

nltk.download('stopwords')

# Init stuff
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def clean_text(text):
    """Basic cleaning: lowercase, remove punctuation, remove stopwords, stem."""
    # Lowercase + strip special characters
    text = re.sub(r'[^\w\s]', '', text.lower())

    # Tokenize
    tokens = text.split()

    # Remove stopwords & stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)

def preprocess_emails(email_list):
    """
    Takes list of dicts with 'body' field.
    Returns cleaned corpus list and TF-IDF matrix + feature names.
    """
    corpus = [clean_text(email.get('body', '')) for email in email_list]

    vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 1),  # flip to (1,2) if you want bigrams
    )
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    return X, feature_names, corpus

def assign_categories(email_list, n_topics = 10):
    '''
    Assign category to the emails
    A list of categories will be returned.
    '''
    X, feature_names, corpus = preprocess_emails(email_list)
    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_

    top_words = [feature_names[topic.argsort()[-1]]for topic in H]
    categories = []
    for weights in W:
        topic_idx = weights.argmax()
        categories.append(top_words[topic_idx])
    return categories

