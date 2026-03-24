# backend/analytics.py

import pandas as pd
from .db import get_connection


# -----------------------
# POPULAR ACTIVITIES
# -----------------------
def get_popular_activities(limit: int = 10):
    try:
        conn = get_connection()

        df = pd.read_sql("""
            SELECT activity_id, COUNT(*) as count
            FROM interactions
            GROUP BY activity_id
            ORDER BY count DESC
            LIMIT ?
        """, conn, params=(limit,))

        # Join with activity details
        activities = pd.read_sql("SELECT * FROM activities", conn)

        conn.close()

        result = []

        for _, row in df.iterrows():
            activity = activities[
                activities["activity_id"] == row["activity_id"]
            ]

            if activity.empty:
                continue

            activity = activity.iloc[0]

            result.append({
                "activity_id": row["activity_id"],
                "title": activity.get("title"),
                "tags": activity.get("tags"),
                "city": activity.get("city"),
                "count": int(row["count"])
            })

        return result

    except Exception as e:
        return {"error": str(e)}


# -----------------------
# USER ANALYTICS
# -----------------------
def get_user_analytics(user_id: str):
    try:
        conn = get_connection()

        interactions = pd.read_sql(
            "SELECT * FROM interactions WHERE user_id = ?",
            conn,
            params=(user_id,)
        )

        activities = pd.read_sql("SELECT * FROM activities", conn)

        conn.close()

        if interactions.empty:
            return {
                "user_id": user_id,
                "total_interactions": 0,
                "top_categories": [],
                "recent_activity": []
            }

        # Total interactions
        total = len(interactions)

        # Merge with activity info
        merged = interactions.merge(
            activities,
            on="activity_id",
            how="left"
        )

        # Extract categories (tags)
        merged["tags"] = merged["tags"].fillna("")

        all_tags = []
        for tags in merged["tags"]:
            all_tags.extend(tags.split(";"))

        tag_series = pd.Series(all_tags)
        top_tags = tag_series.value_counts().head(5).index.tolist()

        # Recent activity
        recent = merged.sort_values("timestamp", ascending=False).head(5)

        recent_list = [
            {
                "activity_id": row["activity_id"],
                "title": row.get("title"),
                "timestamp": row.get("timestamp")
            }
            for _, row in recent.iterrows()
        ]

        return {
            "user_id": user_id,
            "total_interactions": total,
            "top_categories": top_tags,
            "recent_activity": recent_list
        }

    except Exception as e:
        return {"error": str(e)}