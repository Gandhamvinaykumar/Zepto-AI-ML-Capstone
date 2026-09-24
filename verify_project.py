import os
import sqlite3
import pandas as pd
import support_assistant.main as m

print('DB_EXISTS', os.path.exists('data_pipeline/zepto_books.db'))
con = sqlite3.connect('data_pipeline/zepto_books.db')
print('TABLES', con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall())
print('BOOK_COUNT', pd.read_sql_query('SELECT COUNT(*) AS count FROM books', con).to_string(index=False))
print('CAT_COUNT', pd.read_sql_query('SELECT COUNT(*) AS count FROM categories', con).to_string(index=False))
print('TOP_CATS', pd.read_sql_query('SELECT * FROM categories ORDER BY category_id', con).to_string(index=False))
con.close()
print('INTENT_POLICY', m.assistant.classify_intent('What is the delivery fee?'))
print('INTENT_GENERAL', m.assistant.classify_intent('What is the capital of France?'))
chunks = m.assistant.retrieve_chunks('What is the delivery fee?')
print('TOP_CHUNK_ID', chunks[0]['id'])
print('TOP_CHUNK_SNIP', chunks[0]['document'][:120])
