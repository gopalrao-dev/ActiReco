#backend/migrate.py
import pandas as pd
from db import get_connection, init_db

def migrate():
    init_db()
    conn = get_connection()

    # Users
    users = pd.read_csv("data/users.csv")
    users.to_sql("users", conn, if_exists="replace", index=False)

    # Activities
    activities = pd.read_csv("data/activities.csv")
    activities.to_sql("activities", conn, if_exists="replace", index=False)

    # Interactions
    interactions = pd.read_csv("data/interactions.csv")
    interactions.to_sql("interactions", conn, if_exists="replace", index=False)

    conn.close()
    print("✅ Migration complete!")

if __name__ == "__main__":
    migrate()