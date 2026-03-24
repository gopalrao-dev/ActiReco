# backend/db.py
import sqlite3
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "actireco.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # better dict-like access
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # -----------------------
    # USERS TABLE
    # -----------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        name TEXT,
        age INTEGER,
        location TEXT,
        interests TEXT
    )
    """)

    # -----------------------
    # ACTIVITIES TABLE
    # -----------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS activities (
        activity_id TEXT PRIMARY KEY,
        title TEXT,
        tags TEXT,
        city TEXT
    )
    """)

    # -----------------------
    # INTERACTIONS TABLE
    # -----------------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        activity_id TEXT,
        event TEXT,
        rating INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # -----------------------
    # INDEXES (VERY IMPORTANT)
    # -----------------------
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user ON interactions(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity ON interactions(activity_id)")

    conn.commit()
    conn.close()