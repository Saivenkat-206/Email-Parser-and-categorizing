import sqlite3

conn = sqlite3.connect("emails.db")
cur = conn.cursor()
cur.execute("SELECT * FROM emails")
rows = cur.fetchall()
for row in rows:
    print(row)
    print("-" * 40)
