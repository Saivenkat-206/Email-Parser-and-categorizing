# Model Comparison: Random Forest vs NMF for Email Categorization
# This file contains both models with comprehensive evaluation metrics and visualizations

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
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os

nltk.download('stopwords')

# Create plots directory if it doesn't exist
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

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

def plot_confusion_matrix(y_true, y_pred, model_name, class_labels=None):
    '''Generate and save confusion matrix plot'''
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot in plots directory
    filename = os.path.join(PLOTS_DIR, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()
    
    return cm

def plot_metrics_comparison(results, model_names):
    '''Generate and save metrics comparison plot'''
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison: Random Forest vs NMF', 
                 fontsize=16, fontweight='bold', y=1.00)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[model].get(metric, 0) for model in model_names]
        colors = ['#2ecc71', '#3498db']
        
        bars = axes[idx].bar(model_names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[idx].set_ylabel('Score', fontsize=11)
        axes[idx].set_title(label, fontsize=12, fontweight='bold')
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}',
                          ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, 'metrics_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def plot_model_performance_radar(results, model_names):
    '''Generate and save radar chart for model comparison'''
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#2ecc71', '#3498db']
    
    for idx, model in enumerate(model_names):
        values = [results[model].get(metric, 0) for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, 'radar_chart_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def plot_feature_importance(rf_model, feature_names, top_n=15):
    '''Plot top N important features from Random Forest'''
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices], color='#3498db', edgecolor='black', linewidth=1.5)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Most Important Features - Random Forest', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, v in enumerate(importances[indices]):
        plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, 'feature_importance_rf.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def plot_rocauc_curves(y_test, rf_proba, class_labels):
    '''Plot ROC-AUC curves for Random Forest (if binary classification)'''
    if len(np.unique(y_test)) == 2:
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_test, rf_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC-AUC Curve - Random Forest', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        filename = os.path.join(PLOTS_DIR, 'roc_auc_curve.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

def plot_model_comparison_table(results):
    '''Generate and save comparison metrics as a visual table'''
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    comparison_data = []
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for metric_name, metric_key in zip(metrics_names, metric_keys):
        row = [metric_name]
        for model in ['NMF', 'Random Forest']:
            if model in results:
                value = results[model].get(metric_key, 0)
                row.append(f'{value:.4f}')
            else:
                row.append('N/A')
        comparison_data.append(row)
    
    table = ax.table(cellText=comparison_data, 
                    colLabels=['Metric', 'NMF', 'Random Forest'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(comparison_data) + 1):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    plt.title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    filename = os.path.join(PLOTS_DIR, 'metrics_table.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def evaluate_models(email_list, true_labels=None):
    '''
    Evaluate both NMF and Random Forest models with visualization.
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
            unique_labels = sorted(list(set(true_labels)))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            true_labels_numeric = [label_to_idx[label] for label in true_labels]
            nmf_categories_numeric = [label_to_idx.get(cat, 0) for cat in nmf_categories]
        else:
            unique_labels = sorted(list(set(true_labels)))
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
            'predictions': nmf_categories_numeric,
            'cm': confusion_matrix(true_labels_numeric, nmf_categories_numeric)
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
            unique_labels = sorted(list(set(true_labels)))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            y_numeric = [label_to_idx[label] for label in true_labels]
        else:
            unique_labels = sorted(list(set(true_labels)))
            y_numeric = true_labels
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_dense, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
        )
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
        rf.fit(X_train, y_train)
        
        # Make predictions
        rf_predictions = rf.predict(X_test)
        rf_proba = rf.predict_proba(X_test)
        
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
            'predictions': rf_predictions,
            'y_test': y_test,
            'rf_proba': rf_proba,
            'cm': confusion_matrix(y_test, rf_predictions),
            'feature_names': feature_names
        }
    
    # Generate Visualizations
    print("\n" + "="*60)
    print("Generating Visualizations...")
    print("="*60)
    
    if true_labels is not None and 'Random Forest' in results:
        # Comparison metrics table
        print("\n" + "-"*60)
        print("Model Comparison Summary")
        print("-"*60)
        
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
        print()
        
        # Plot confusion matrices
        plot_confusion_matrix(results['NMF']['predictions'], 
                            results['NMF']['predictions'], 
                            'NMF', 
                            unique_labels)
        
        plot_confusion_matrix(results['Random Forest']['y_test'], 
                            results['Random Forest']['predictions'], 
                            'Random Forest', 
                            unique_labels)
        
        # Plot metrics comparison
        plot_metrics_comparison(results, ['NMF', 'Random Forest'])
        
        # Plot metrics table
        plot_model_comparison_table(results)
        
        # Plot radar chart
        plot_model_performance_radar(results, ['NMF', 'Random Forest'])
        
        # Plot feature importance (only for RF)
        plot_feature_importance(results['Random Forest']['model'], 
                              results['Random Forest']['feature_names'], 
                              top_n=15)
        
        # Determine better model
        print("\n" + "-"*60)
        if results['Random Forest']['f1_score'] > results['NMF']['f1_score']:
            print("✓ Random Forest performs better (higher F1-Score)")
            print(f"  Improvement: {(results['Random Forest']['f1_score'] - results['NMF']['f1_score'])*100:.2f}%")
        else:
            print("✓ NMF performs better or equally (higher F1-Score)")
            print(f"  Difference: {(results['NMF']['f1_score'] - results['Random Forest']['f1_score'])*100:.2f}%")
        
        print("\n" + "="*60)
        print(f"All plots saved to '{PLOTS_DIR}/' directory!")
        print("="*60)
    
    return results

def print_detailed_report(y_true, y_pred, model_name):
    '''Print detailed classification report'''
    print(f"\nDetailed Classification Report for {model_name}:")
    print(classification_report(y_true, y_pred))

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
        {'body': 'meeting notes from yesterday team discussion'},
        {'body': 'billing statement for current month services'},
        {'body': 'special promotional offer valid until end of week'},
        {'body': 'team building event next month rsvp required'},
        {'body': 'contract renewal reminder action needed soon'},
        {'body': 'bug fix release addresses critical vulnerability'},
        {'body': 'customer support ticket needs immediate attention'},
        {'body': 'budget approval required for q4 expenses'},
        {'body': 'office party celebration this friday'},
        {'body': 'software update improves user experience'},
    ]
    
    # Sample labels (Work, Billing, Marketing, Social, Finance, Tech, Support, Finance, Tech, Social)
    sample_labels = ['work', 'billing', 'work', 'marketing', 'social', 
                     'billing', 'tech', 'support', 'finance', 'tech',
                     'work', 'billing', 'marketing', 'social', 'work',
                     'tech', 'support', 'finance', 'social', 'tech']
    
    print("Running Model Comparison with Visualizations...")
    results = evaluate_models(sample_emails, sample_labels)
    print("\n✓ Evaluation Complete!")
