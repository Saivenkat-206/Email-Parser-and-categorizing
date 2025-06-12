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
If that's a bit too much work, just download the zip and extract it
### 2. Create .env file
fill the .env file with
```
IMAP_SERVER='imap.google or outlook.com' # if you use google - gmail else use outlook (basically your domain name)
EMAIL_USER=you@example.com
EMAIL_PASS=your_email_password_or_app_pass
```
Fill your creds in place of the emailID and Password
#### Filling the creds:
### For Google:
1. Go to Gmail Settings and enable IMAP access under Forwarding and POP/IMAP.
2. Login if there is any 2FA enabled.
3. Now go to account -> security.
4. Under signing in to google select app passwords.
5. Choose mail -> other and give it a name if your choice.
6. Copy the 16 character password and paste it.
#### This is because your password is not secure enough, this setup is necessary
### For Outlook:
1. Check outlook settings.
2. Go to microsoft security basics.
3. Under security, click Add a new way to sign in.
4. Choose app passwords and generate one for IMAP access.
### 3. Setup a virtual environment (optinal)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 4. Install dependencies
```bash
pip install -r requirements.txt
```
### 5. Finally run the app:
```bash
streamlit run Dashboard.py
```
## NOTE:
This is an unsupervised model, so the "departments" are inferred from word distributions. You can fine-tune the model or replace the logic with your own labeled data if needed

