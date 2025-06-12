import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import main
import sqlite3
import os

st.set_page_config(layout = "wide")
st.title("Email Dashboard")

DB_FILE = main.DB_FILE
max_emails = st.number_input("Number of emails to download", min_value=1, max_value=1000, value=100, step=1)
def run_email_download():
    os.system(f'python main.py {max_emails}')

def load_emails():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM emails", conn)
    conn.close()
    return df

if st.button("Download Emails"):
    with st.spinner("Downloading emails..."):
        run_email_download()
    st.success("Emails downloaded successfully!")

df = load_emails()
if df.empty:
    st.warning("No emails found in the database. Click on download emails to download your emails")
    st.stop()

st.header("Email data")
st.dataframe(df)

st.header("Email stats")
cat_count = df['Category'].value_counts()
fig, ax = plt.subplots(figsize=(6, 3))
sns.barplot(x=cat_count.index, y=cat_count.values, ax=ax)
ax.set_title("Email Categories Counting Plot")
ax.set_xlabel("Category")
ax.set_ylabel("Count")
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig)

st.header("Important words by Category")
cats = df['Category'].unique()
selected_cat = st.selectbox("Select a category",cats)
cat_bodies = ''.join(df[df['Category'] == selected_cat]['Body'].dropna())
if cat_bodies.strip():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cat_bodies)
    fig_wc, ax_wc = plt.subplots(figsize=(6, 3))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
else:
    st.info(f"No content available for category: {selected_cat}")

st.header("attachments based of category")
selected_cat = st.selectbox("Select a category to see attachments", cats, key='attach')
attachments = df[(df['Category'] == selected_cat) & (df['Attachment_Path'].notnull())]

if not attachments.empty:
    for i, r in attachments.iterrows():
        if r['Attachment_Path']:
            paths = [p.strip() for p in r['Attachment_Path'].split(',')]
            for path in paths:
                if os.path.exists(path):
                    filename = os.path.basename(path)
                    with open(path, 'rb') as f:
                        st.download_button(
                            label = f'Download {filename}',
                            data = f,
                            file_name = filename,
                            mime = 'application/octet-stream'
                        )
else:
    st.info(f'No attachments found for category: {selected_cat}')

