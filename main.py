# Import Packages
import imaplib as imp
import email
from email.header import decode_header
import os
import _sqlite3
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import logging
from NLP import assign_categories

# Set up logs and activities
logging.basicConfig(level=logging.INFO, format='%(asctime)s, - %(name)s, - %(message)s') 

# Email settings stuff and the storage part
load_dotenv()

# Initialize the credentials
EMAIL = os.getenv('EMAIL')
PASSWORD = os.getenv('PASSWORD')
IMAP_SERVER = 'imap.gmail.com'
ATTACHMENT_DIR = 'attachments'
DB_FILE = 'emails.db'

# Create dir to store the stuff
os.makedirs(ATTACHMENT_DIR, exist_ok = True)

#initialize SQLlite database

def _init_db():
    conn = _sqlite3.connect(DB_FILE) # connect to file
    curr = conn.cursor() # initialize curson object

    #Email table
    curr.execute('''
        CREATE TABLE IF NOT EXISTS emails(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 Subject STRING,
                 Sender STRING,
                 Date STRING,
                 Category STRING,
                 Body STRING,
                 Attachment_Path STRING)
    ''' 
    )

    #Data table
    curr.execute('''
        CREATE TABLE IF NOT EXISTS extracted_data(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id INTEGER,
                field_name TEXT,
                field_value TEXT,
                FOREIGN KEY (email_id) REFERENCES emails (id))
    '''
    )
    conn.commit()
    return conn

# Email Server Connection
def connect_to_email():
    mail = imp.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select('inbox')
    return mail

#Fetch the Email data
def fetch_emails(mail, conn, max_emails=10):
    status, messages = mail.search(None, 'ALL')
    if status != 'OK':
        logging.error('Failed to reach Emails')
        return

    email_ids = messages[0].split()
    email_ids = email_ids[-max_emails:]

    curr = conn.cursor()

    for eid in email_ids:
        res, msg_data = mail.fetch(eid, '(RFC822)')
        if res != 'OK':
            logging.warning(f'error fetching email: {eid}')
            continue

        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        subject = decode_field(msg.get('Subject'))
        sender = decode_field(msg.get('From'))
        date = msg.get('Date')
        body = get_body(msg)

        # Skip if already in DB
        curr.execute("SELECT COUNT(*) FROM emails WHERE Subject = ? AND Sender = ? AND Date = ?", (subject, sender, date))
        if curr.fetchone()[0] > 0:
            continue

        logging.info(f'Email from {sender} | Subject: {subject}')

        category = "Uncategorized"
        attachment_path = None

        curr.execute('''
            INSERT INTO emails (Subject, Sender, Date, Category, Body, Attachment_Path)
            VALUES(?, ?, ?, ?, ?, ?)
        ''', (subject, sender, date, category, body, attachment_path))

    conn.commit()

# The helper functions for decoding subject, sender address and getting body
def decode_field(field):
    if field is None:
        return ""
    parts = decode_header(field) # Split the field into the text and it's encoding 
    decoded = '' # This string will be returned as result
    for part, encoding in parts:
        if isinstance(part, bytes): # If a part is in bytes 
            decoded += part.decode(encoding or 'utf-8', errors='ignore') # Default encoding is utf-8 so this will converted to simple text
        else:
            decoded += part # Ff it already is simple then nice
    return decoded

def get_body(msg):
    body = ""
    if msg.is_multipart(): # If the message has multiple parts
        for part in msg.walk(): # Iterate thru the message
            content_type = part.get_content_type()  
            content_dispo = str(part.get('Content-Disposition')) # The part gets ignored
            if 'attachment' in content_dispo:
                continue
            payload = part.get_payload(decode=True) # Gets the actual content
            if payload:
                if content_type == 'text/plain': # chumma decode and return
                    return payload.decode(errors='ignore')  # Prefer plain text
                elif content_type == 'text/html' and not body:
                    # Fallback if no plain text, save HTML for later
                    body = BeautifulSoup(payload.decode(errors='ignore'), 'html.parser').get_text()
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(errors='ignore')
    return body

def save_attachments(msg, email_id):
    attachment_paths = []
    for part in msg.walk():
        content_dispo = str(part.get("Content-Disposition"))
        if "attachment" in content_dispo: # If an attachment type stuff is found in the email
            filename = part.get_filename() # Get the name of the file
            if filename:
                decoded_name = decode_field(filename) 
                filepath = os.path.join(ATTACHMENT_DIR, f"{email_id}_{decoded_name}") # The name of the file will be the mail id and the decoded name 
                with open(filepath, "wb") as f:
                    f.write(part.get_payload(decode=True)) # Write the attachment in wb mode
                attachment_paths.append(filepath)
    return ', '.join(attachment_paths) if attachment_paths else None

# Retreive emails from database where the category is unknown 
def get_emails_for_categorization(conn):
    curr = conn.cursor()
    curr.execute('SELECT id, Body FROM emails WHERE Category = "Uncategorized"')
    return [{'id': row[0], 'body': row[1]} for row in curr.fetchall()]

if __name__ == '__main__':
    print("Starting...")
    conn = _init_db()
    mail = connect_to_email()
    fetch_emails(mail, conn)
    mail.logout()
    emails = get_emails_for_categorization(conn)

    if emails:
        categories = assign_categories(emails)
        curr = conn.cursor()
        for email, category in zip(emails, categories):
            curr.execute('UPDATE emails SET Category = ? WHERE id = ?', (category, email['id']))
        conn.commit()

    conn.close()
    print("Done")

