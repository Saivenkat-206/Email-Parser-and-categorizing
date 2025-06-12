# Import Packages
import imaplib as imp
import email
from email.header import decode_header
import os
import sqlite3
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

#initialize SQLlite databases
def _init_db():
    conn = sqlite3.connect(DB_FILE) # connect to file
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
    conn.commit()
    return conn

# Email Server Connection
def connect_to_email():
    mail = imp.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select('inbox')
    return mail

#Fetch the Email data
def fetch_emails(mail, conn, max_emails):
    status, messages = mail.search(None, 'ALL')
    if status != 'OK':
        logging.error('Failed to reach Emails')
        return 0

    email_ids = messages[0].split()
    email_ids = email_ids[-max_emails:]
    curr = conn.cursor()
    new_emails_added = 0

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
        curr.execute("SELECT COUNT(*) FROM emails WHERE Subject = ? AND Sender = ? AND Date = ?", (subject, sender, date))
        
        if curr.fetchone()[0] > 0:
            continue
        logging.info(f'Email from {sender} | Subject: {subject}')
        category = "Uncategorized"
        # Save attachments and store their paths
        attachment_path = save_attachments(msg, eid.decode() if isinstance(eid, bytes) else str(eid))
        curr.execute('''
            INSERT INTO emails (Subject, Sender, Date, Category, Body, Attachment_Path)
            VALUES(?, ?, ?, ?, ?, ?)
        ''', (subject, sender, date, category, body, attachment_path))
        new_emails_added += 1

    conn.commit()
    return new_emails_added

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
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_dispo = str(part.get('Content-Disposition'))
            if 'attachment' in content_dispo:
                continue
            payload = part.get_payload(decode=True)
            if payload:
                text = payload.decode(errors='ignore')
                # Always strip HTML tags, even for plain text
                if content_type == 'text/html':
                    body = BeautifulSoup(text, 'html.parser').get_text()
                elif content_type == 'text/plain' and not body:
                    body = BeautifulSoup(text, 'html.parser').get_text()
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            text = payload.decode(errors='ignore')
            body = BeautifulSoup(text, 'html.parser').get_text()
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
    import sys
    print("Starting...")
    conn = _init_db()
    mail = connect_to_email()
    # Allow number of emails to fetch as argument
    max_emails = 100
    if len(sys.argv) > 1:
        try:
            max_emails = int(sys.argv[1])
        except Exception:
            pass

    total_needed = max_emails
    total_added = 0
    attempts = 0
    max_attempts = 20
    checked_email_ids = set()

    while total_added < total_needed and attempts < max_attempts:
        # Fetch all email IDs from the server
        status, messages = mail.search(None, 'ALL')
        if status != 'OK':
            logging.error('Failed to reach Emails')
            break

        all_email_ids = messages[0].split()
        # Filter out email IDs that have already been checked
        email_ids_to_check = [eid for eid in reversed(all_email_ids) if eid not in checked_email_ids]
        # Only check up to the number needed
        email_ids_to_fetch = email_ids_to_check[:total_needed - total_added]

        if not email_ids_to_fetch:
            logging.info("No more unique emails to fetch from server.")
            break
        curr = conn.cursor()
        new_emails_added = 0

        for eid in email_ids_to_fetch:
            checked_email_ids.add(eid)
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
            curr.execute("SELECT COUNT(*) FROM emails WHERE Subject = ? AND Sender = ? AND Date = ?", (subject, sender, date))
            if curr.fetchone()[0] > 0:
                continue

            logging.info(f'Email from {sender} | Subject: {subject}')

            category = "Uncategorized"
            # FIX: Save attachments and store their paths
            attachment_path = save_attachments(msg, eid.decode() if isinstance(eid, bytes) else str(eid))
            curr.execute('''
                INSERT INTO emails (Subject, Sender, Date, Category, Body, Attachment_Path)
                VALUES(?, ?, ?, ?, ?, ?)
            ''', (subject, sender, date, category, body, attachment_path))
            new_emails_added += 1

        conn.commit()
        total_added += new_emails_added
        attempts += 1
        # If no new emails were added in this batch, try the older emails
        if new_emails_added == 0:
            logging.info("No new unique emails found in this batch, trying next set...")
            continue

    mail.logout()
    emails = get_emails_for_categorization(conn)
    if emails:
        categories = assign_categories(emails)
        curr = conn.cursor()
        for email, category in zip(emails, categories):
            curr.execute('UPDATE emails SET Category = ? WHERE id = ?', (category, email['id']))
        conn.commit()
    conn.close()
    print(f"Done. {total_added} new emails added.")

