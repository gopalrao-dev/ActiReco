from .db import get_connection
import pandas as pd


def get_user_interactions(user_id):
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM interactions WHERE user_id = ?",
        conn,
        params=(user_id,)
    )
    conn.close()
    return df


def get_popular_activities():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT activity_id, COUNT(*) as cnt
        FROM interactions
        GROUP BY activity_id
        ORDER BY cnt DESC
    """, conn)
    conn.close()
    return df