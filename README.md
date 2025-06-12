# Email Categorization Dashboard
This project automatically pulls emails, analyzes them using NLP (Non-Negative Matrix Factorization), and categorizes them into department-like topics. It also gives you a Streamlit dashboard to visualize trends, keyword importance, and category distribution.

# Features
- Pulls and processes emails (body text)  
- Cleans and vectorizes using TF-IDF  
- Extracts topics using NMF (unsupervised learning)  
- Assigns emails to the closest department/topic  
- Displays everything in a clean interactive dashboard

## Setup

### 1. Clone the project
```bash
git clone https://github.com/Saivenkat-206/Email-Parser-and-categorizing.git
cd Email-Parser-and-categorizing
```
### 2. Create .env file
fill the .env file with
```
EMAIL_USER=you@example.com
EMAIL_PASS=your_email_password_or_app_pass
```
### 3. Setup a virtual environment (optinal)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 4. Install dependencies
```bash
pip install -r requirements.txt
```

Finally run the app:
```bash
streamlit run Dashboard.py
```

