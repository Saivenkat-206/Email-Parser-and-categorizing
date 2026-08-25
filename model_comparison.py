# Model Comparison: Random Forest vs NMF for Email Categorization
# This file contains both models with comprehensive evaluation metrics

import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)
import numpy as np
import pandas as pd

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

def assign_categories_nmf(email_list, n_topics=10):
    '''
    Assign category to the emails using NMF (Non-Negative Matrix Factorization)
    A list of categories will be returned.
    '''
    X, feature_names, corpus = preprocess_emails(email_list)
    nmf = NMF(n_components=n_topics, random_state=42, init='random', max_iter=500)
    W = nmf.fit_transform(X)
    H = nmf.components_
    
    top_words = [feature_names[topic.argsort()[-1]] for topic in H]
    categories = []
    for weights in W:
        topic_idx = weights.argmax()
        categories.append(top_words[topic_idx])
    return categories, nmf, X, feature_names

def assign_categories_rf(email_list, y_labels=None):
    '''
    Assign category to the emails using Random Forest Classifier.
    If y_labels is provided, trains on labeled data and returns predictions.
    If y_labels is None, returns None (requires labeled data).
    '''
    X, feature_names, corpus = preprocess_emails(email_list)
    
    if y_labels is None:
        print("Warning: Random Forest requires labeled training data. Provide y_labels for training.")
        return None, None, X, feature_names
    
    # Convert sparse matrix to dense for Random Forest
    X_dense = X.toarray()
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_dense, y_labels)
    
    # Get predictions
    predictions = rf.predict(X_dense)
    return predictions, rf, X, feature_names

def evaluate_models(email_list, true_labels=None):
    '''
    Evaluate both NMF and Random Forest models.
    For Random Forest, true_labels must be provided.
    '''
    results = {}
    
    # NMF Model Evaluation
    print("\n" + "="*60)
    print("NMF (Non-Negative Matrix Factorization) Model")
    print("="*60)
    
    nmf_categories, nmf_model, X, feature_names = assign_categories_nmf(email_list, n_topics=10)
    
    if true_labels is not None:
        # Convert categorical labels to numeric if needed
        if isinstance(true_labels[0], str):
            unique_labels = list(set(true_labels))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            true_labels_numeric = [label_to_idx[label] for label in true_labels]
            nmf_categories_numeric = [label_to_idx.get(cat, 0) for cat in nmf_categories]
        else:
            true_labels_numeric = true_labels
            nmf_categories_numeric = nmf_categories
        
        nmf_accuracy = accuracy_score(true_labels_numeric, nmf_categories_numeric)
        nmf_precision = precision_score(true_labels_numeric, nmf_categories_numeric, average='weighted', zero_division=0)
        nmf_recall = recall_score(true_labels_numeric, nmf_categories_numeric, average='weighted', zero_division=0)
        nmf_f1 = f1_score(true_labels_numeric, nmf_categories_numeric, average='weighted', zero_division=0)
        
        print(f"Accuracy:  {nmf_accuracy:.4f}")
        print(f"Precision: {nmf_precision:.4f}")
        print(f"Recall:    {nmf_recall:.4f}")
        print(f"F1-Score:  {nmf_f1:.4f}")
        
        results['NMF'] = {
            'accuracy': nmf_accuracy,
            'precision': nmf_precision,
            'recall': nmf_recall,
            'f1_score': nmf_f1,
            'model': nmf_model,
            'predictions': nmf_categories
        }
    else:
        print("NMF is unsupervised - cannot compute supervised metrics without true labels.")
        print("NMF Topics assigned successfully.")
        results['NMF'] = {
            'model': nmf_model,
            'predictions': nmf_categories
        }
    
    # Random Forest Model Evaluation
    print("\n" + "="*60)
    print("Random Forest Classifier Model")
    print("="*60)
    
    if true_labels is None:
        print("Random Forest requires labeled training data.")
        print("Provide true_labels to evaluate the Random Forest model.")
    else:
        # Split data for training and testing
        X_dense = X.toarray()
        
        if isinstance(true_labels[0], str):
            unique_labels = list(set(true_labels))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            y_numeric = [label_to_idx[label] for label in true_labels]
        else:
            y_numeric = true_labels
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_dense, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
        )
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
        rf.fit(X_train, y_train)
        
        # Make predictions
        rf_predictions = rf.predict(X_test)
        
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        rf_precision = precision_score(y_test, rf_predictions, average='weighted', zero_division=0)
        rf_recall = recall_score(y_test, rf_predictions, average='weighted', zero_division=0)
        rf_f1 = f1_score(y_test, rf_predictions, average='weighted', zero_division=0)
        
        print(f"Accuracy:  {rf_accuracy:.4f}")
        print(f"Precision: {rf_precision:.4f}")
        print(f"Recall:    {rf_recall:.4f}")
        print(f"F1-Score:  {rf_f1:.4f}")
        
        results['Random Forest'] = {
            'accuracy': rf_accuracy,
            'precision': rf_precision,
            'recall': rf_recall,
            'f1_score': rf_f1,
            'model': rf,
            'predictions': rf_predictions
        }
    
    # Model Comparison
    if true_labels is not None and 'Random Forest' in results:
        print("\n" + "="*60)
        print("Model Comparison Summary")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            'NMF': [
                results['NMF']['accuracy'],
                results['NMF']['precision'],
                results['NMF']['recall'],
                results['NMF']['f1_score']
            ],
            'Random Forest': [
                results['Random Forest']['accuracy'],
                results['Random Forest']['precision'],
                results['Random Forest']['recall'],
                results['Random Forest']['f1_score']
            ]
        }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        
        print(comparison_df.to_string())
        
        # Determine better model
        print("\n" + "-"*60)
        if results['Random Forest']['f1_score'] > results['NMF']['f1_score']:
            print("✓ Random Forest performs better (higher F1-Score)")
            print(f"  Improvement: {(results['Random Forest']['f1_score'] - results['NMF']['f1_score'])*100:.2f}%")
        else:
            print("✓ NMF performs better or equally (higher F1-Score)")
            print(f"  Difference: {(results['NMF']['f1_score'] - results['Random Forest']['f1_score'])*100:.2f}%")
    
    return results

def print_detailed_report(true_labels, predictions, model_name):
    '''Print detailed classification report'''
    print(f"\nDetailed Classification Report for {model_name}:")
    print(classification_report(true_labels, predictions))

# Example usage
if __name__ == "__main__":
    # Example: Create dummy labeled email data for testing
    sample_emails = [
        {'body': 'urgent meeting scheduled for tomorrow morning at 9am conference room'},
        {'body': 'your account has been charged $99 for subscription renewal'},
        {'body': 'project deadline extended to next friday please update team'},
        {'body': 'welcome bonus and free trial offer limited time only'},
        {'body': 'team lunch tomorrow catering from italian restaurant'},
        {'body': 'invoice payment due within 30 days please remit immediately'},
        {'body': 'new feature release improves performance and security'},
        {'body': 'customer complaint about product quality needs resolution'},
        {'body': 'quarterly earnings report and financial projections attached'},
        {'body': 'system maintenance scheduled for sunday evening downtime expected'},
    ]
    
    # Sample labels (Work, Billing, Marketing, Social, Finance, Tech, Support, Finance, Tech, Social)
    sample_labels = ['work', 'billing', 'work', 'marketing', 'social', 
                     'billing', 'tech', 'support', 'finance', 'tech']
    
    print("Running Model Comparison...")
    results = evaluate_models(sample_emails, sample_labels)
    print("\n✓ Evaluation Complete!")
