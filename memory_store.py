# memory_store.py
#
# Replaces the old memory.py's "keep last 40 raw messages" model.
#
# Three separate stores instead of one growing transcript:
#   - profile.json      durable facts about the person (name, age, job...)
#   - memory.db          SQLite: memory_events (time-scoped facts to follow
#                         up on) + topics (recurring interests, weighted by
#                         mention frequency + recency)
#   - working_turns.json a small rolling buffer (last few exchanges only,
#                         for immediate coherence, not long-term recall)
#
# The goal: the LLM's context stays flat-sized no matter how long the
# relationship goes on. Long-term "memory" comes from compact facts/events/
# topics, not from re-reading old conversation.

import json
import os
import random
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "memory.db")
PROFILE_FILE = os.path.join(BASE_DIR, "profile.json")
WORKING_TURNS_FILE = os.path.join(BASE_DIR, "working_turns.json")

WORKING_TURNS_MAX = 6  # ~3 exchanges of immediate context

TOPIC_MIN_MENTIONS = 2       # don't treat a one-off mention as a "recurring interest"
TOPIC_QUOTES_MAX = 3


# ============================================================
# SETUP
# ============================================================

def _connect():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(_connect()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                category TEXT,
                created_at TEXT NOT NULL,
                recall_after TEXT NOT NULL,
                recall_expire_at TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                source_quote TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL UNIQUE,
                mention_count INTEGER NOT NULL DEFAULT 1,
                first_mentioned_at TEXT NOT NULL,
                last_mentioned_at TEXT NOT NULL,
                example_quotes TEXT
            )
        """)
        conn.commit()


init_db()


# ============================================================
# PROFILE  (durable facts: name, age, location, job, etc.)
# ============================================================

def get_profile():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as f:
            return json.load(f)
    return {}


def update_profile(deltas):
    """Merge new/updated facts into the profile. Empty/None values ignored."""
    if not deltas:
        return get_profile()

    profile = get_profile()
    for key, value in deltas.items():
        if value in (None, "", []):
            continue
        profile[key] = value

    with open(PROFILE_FILE, "w") as f:
        json.dump(profile, f, indent=2)

    return profile


# ============================================================
# WORKING TURNS  (short rolling buffer, NOT long-term memory)
# ============================================================

def get_working_turns():
    if os.path.exists(WORKING_TURNS_FILE):
        with open(WORKING_TURNS_FILE, "r") as f:
            return json.load(f)
    return []


def append_working_turn(role, content):
    turns = get_working_turns()
    turns.append({"role": role, "content": content})
    turns = turns[-WORKING_TURNS_MAX:]

    with open(WORKING_TURNS_FILE, "w") as f:
        json.dump(turns, f, indent=2)

    return turns


# ============================================================
# MEMORY EVENTS  (time-scoped facts: "holiday in 3 weeks")
# ============================================================

def add_event(text, category=None, recall_after_days=1, recall_expire_days=None, source_quote=None):
    now = datetime.now()
    recall_after = now + timedelta(days=recall_after_days)
    recall_expire_at = (
        now + timedelta(days=recall_after_days + recall_expire_days)
        if recall_expire_days is not None
        else None
    )

    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO memory_events
                (text, category, created_at, recall_after, recall_expire_at, status, source_quote)
            VALUES (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                text,
                category,
                now.isoformat(),
                recall_after.isoformat(),
                recall_expire_at.isoformat() if recall_expire_at else None,
                source_quote,
            ),
        )
        conn.commit()


def get_due_events(limit=1):
    """Pending events whose recall window is currently open."""
    now = datetime.now().isoformat()

    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT * FROM memory_events
            WHERE status = 'pending'
              AND recall_after <= ?
              AND (recall_expire_at IS NULL OR recall_expire_at >= ?)
            ORDER BY recall_after ASC
            LIMIT ?
            """,
            (now, now, limit),
        ).fetchall()

    return [dict(r) for r in rows]


def resolve_event(event_id, status="resolved"):
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE memory_events SET status = ? WHERE id = ?",
            (status, event_id),
        )
        conn.commit()


def mark_event_mentioned(event_id):
    resolve_event(event_id, status="mentioned")


# ============================================================
# TOPICS  (recurring interests, weighted by frequency + recency)
# ============================================================

def touch_topic(label, quote=None):
    """Record a mention of a recurring topic. Upserts + bumps mention_count."""
    label = label.strip().lower()
    if not label:
        return

    now = datetime.now().isoformat()

    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM topics WHERE label = ?", (label,)
        ).fetchone()

        if row is None:
            quotes = [quote] if quote else []
            conn.execute(
                """
                INSERT INTO topics
                    (label, mention_count, first_mentioned_at, last_mentioned_at, example_quotes)
                VALUES (?, 1, ?, ?, ?)
                """,
                (label, now, now, json.dumps(quotes)),
            )
        else:
            quotes = json.loads(row["example_quotes"] or "[]")
            if quote and quote not in quotes:
                quotes.append(quote)
            quotes = quotes[-TOPIC_QUOTES_MAX:]

            conn.execute(
                """
                UPDATE topics
                SET mention_count = mention_count + 1,
                    last_mentioned_at = ?,
                    example_quotes = ?
                WHERE id = ?
                """,
                (now, json.dumps(quotes), row["id"]),
            )

        conn.commit()


def _effective_weight(mention_count, last_mentioned_at):
    """Recency-decayed frequency score. Recent + frequent = high weight."""
    try:
        last = datetime.fromisoformat(last_mentioned_at)
        days_since = max((datetime.now() - last).total_seconds() / 86400.0, 0)
    except (TypeError, ValueError):
        days_since = 999
    return mention_count / (1.0 + days_since)


def get_weighted_topic(exclude_labels=None):
    """Pick one recurring topic to potentially bring up, weighted-random by
    (frequency, recency). Returns None if nothing qualifies."""
    exclude_labels = {l.lower() for l in (exclude_labels or [])}

    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM topics WHERE mention_count >= ?",
            (TOPIC_MIN_MENTIONS,),
        ).fetchall()

    candidates = [dict(r) for r in rows if r["label"] not in exclude_labels]
    if not candidates:
        return None

    weights = [_effective_weight(c["mention_count"], c["last_mentioned_at"]) for c in candidates]
    if sum(weights) <= 0:
        return None

    return random.choices(candidates, weights=weights, k=1)[0]


# ============================================================
# CONTEXT ASSEMBLY
# ============================================================

def build_memory_context(session_mentioned_topics=None):
    """
    Returns a compact block to inject into the system prompt: known
    profile facts + at most one thing worth naturally raising (a due
    event takes priority over a recurring-topic suggestion).
    """
    lines = []

    profile = get_profile()
    if profile:
        lines.append("Things you know about this person:")
        for key, value in profile.items():
            lines.append(f"- {key}: {value}")

    due = get_due_events(limit=1)
    suggestion = None

    if due:
        event = due[0]
        suggestion = (
            f"There's something you could naturally ask about if it fits: "
            f"{event['text']}. Don't force it into the first message, but if "
            f"a natural opening comes up, ask about it. (internal id: {event['id']})"
        )
    else:
        topic = get_weighted_topic(exclude_labels=session_mentioned_topics)
        if topic:
            suggestion = (
                f"This person often talks about \"{topic['label']}\". If the "
                f"conversation is a little quiet or you're looking for something "
                f"to bring up, you could naturally steer toward that — but only "
                f"if it fits, don't force it."
            )

    if suggestion:
        lines.append("")
        lines.append(suggestion)

    return "\n".join(lines), (due[0]["id"] if due else None)
