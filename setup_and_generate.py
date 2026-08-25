#!/usr/bin/env python3
"""
Quick setup script to generate model comparison plots
Run this locally: python setup_and_generate.py
"""

import subprocess
import sys
import os

print("="*60)
print("Setting up and generating model comparison plots")
print("="*60)

# Install required packages
print("\n📦 Installing required packages...")
packages = [
    'matplotlib',
    'seaborn', 
    'pandas',
    'scikit-learn',
    'nltk',
    'beautifulsoup4',
    'python-dotenv'
]

for package in packages:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
        print(f"✓ {package} installed")
    except Exception as e:
        print(f"⚠ Error installing {package}: {e}")

print("\n🚀 Running model comparison...")
try:
    subprocess.check_call([sys.executable, 'run_model_comparison.py'])
    print("\n✓ Plots generated successfully!")
    print("📁 Check the 'plots/' directory for all generated visualizations")
except Exception as e:
    print(f"❌ Error running model comparison: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ All done! Your plots are ready in the 'plots/' folder")
print("="*60)
