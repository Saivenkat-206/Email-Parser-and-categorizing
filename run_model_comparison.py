#!/usr/bin/env python3
"""
Standalone script to run model comparison and generate plots.
This script executes the model comparison with sample data and saves plots.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_comparison import evaluate_models

# Sample email data
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

# Sample labels
sample_labels = ['work', 'billing', 'work', 'marketing', 'social', 
                 'billing', 'tech', 'support', 'finance', 'tech',
                 'work', 'billing', 'marketing', 'social', 'work',
                 'tech', 'support', 'finance', 'social', 'tech']

if __name__ == "__main__":
    print("="*60)
    print("Running Model Comparison with Sample Data")
    print("="*60)
    results = evaluate_models(sample_emails, sample_labels)
    print("\n✓ Model Comparison Complete!")
    print("All plots saved to 'plots/' directory")
