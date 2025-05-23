# https://grok.com/chat/7c6c7c4c-9e17-43a1-916e-cc5aee4c9cfd
import sqlite3
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# 初始化数据库papers.db
def init_db():
    with sqlite3.connect('papers.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,
                affiliation TEXT,
                date TEXT,
                abstract TEXT,
                introduction TEXT,
                funding TEXT,
                conclusion TEXT,
                created_at TIMESTAMP
            )
        ''')
        conn.commit()

# 插入数据
def insert_paper(data):
    with sqlite3.connect('papers.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO papers (title, authors, affiliation, date, abstract, introduction, funding, conclusion, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('title', ''),
            data.get('authors', ''),
            data.get('affiliation', ''),
            data.get('date', ''),
            data.get('abstract', ''),
            data.get('introduction', ''),
            data.get('funding', ''),
            data.get('conclusion', ''),
            datetime.now().isoformat()
        ))
        conn.commit()
        return cursor.lastrowid

@app.route('/')
def serve_form():
    return app.send_static_file('form.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    try:
        data = request.form
        if not data.get('title') or not data.get('authors'):
            return jsonify({'error': '标题和作者为必填项'}), 400
        paper_id = insert_paper(data)
        return jsonify({'message': f'数据已保存，ID: {paper_id}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8848, debug=True)