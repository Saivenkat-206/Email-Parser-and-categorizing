import sqlite3

conn = sqlite3.connect("emails.db")
cur = conn.cursor()
cur.execute("SELECT Category FROM emails")
rows = cur.fetchall()
for row in rows:
    print(row[0])
    print("-" * 40)
