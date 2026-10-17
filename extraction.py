# extraction.py
#
# After each reply is sent to the user, run one extra small LLM call in
# the background to pull structured memory updates out of that exchange:
# profile facts, new time-scoped events, resolved events, and recurring
# topics. This runs AFTER the user-facing reply, in a background thread,
# so it never adds to conversational latency.

import json
import re
import threading

from openai import OpenAI

from config import API_BASE, MODEL, API_KEY
import memory_store

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

EXTRACTION_PROMPT = """You extract structured memory updates from one exchange \
between a user and an AI companion. Reply with ONLY a JSON object, no other text.

JSON shape:
{{
  "profile_updates": {{}},        // durable facts about the user, e.g. {{"job": "electrician"}}. Empty object if none.
  "new_events": [],               // future-relevant plans/events. Each: {{"text": str, "category": str, "recall_after_days": int, "recall_expire_days": int or null}}
  "resolved_event_ids": [],       // ints, from "Currently tracked" list below, if this exchange already covered them
  "topics": []                    // short lowercase labels for recurring-interest topics mentioned this turn, e.g. ["cars", "job search"]
}}

Only extract things actually stated. Don't invent details. If nothing is
worth recording, return empty values for everything.

Currently tracked pending events:
{pending}

Exchange:
User: {user_msg}
Assistant: {assistant_reply}

JSON:"""


def _extract_json(text):
    """Local models don't always emit clean JSON. Find the first {...} block."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _run_extraction(user_msg, assistant_reply):
    pending = memory_store.get_due_events(limit=10)
    pending_desc = "\n".join(f"- id={e['id']}: {e['text']}" for e in pending) or "(none)"

    prompt = EXTRACTION_PROMPT.format(
        pending=pending_desc,
        user_msg=user_msg,
        assistant_reply=assistant_reply,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[extraction] LLM call failed: {e}")
        return

    data = _extract_json(raw)
    if data is None:
        print(f"[extraction] could not parse JSON from: {raw[:200]}")
        return

    try:
        profile_updates = data.get("profile_updates") or {}
        if profile_updates:
            memory_store.update_profile(profile_updates)

        for event in data.get("new_events") or []:
            memory_store.add_event(
                text=event.get("text", "").strip(),
                category=event.get("category"),
                recall_after_days=event.get("recall_after_days", 1),
                recall_expire_days=event.get("recall_expire_days"),
                source_quote=user_msg,
            )

        for event_id in data.get("resolved_event_ids") or []:
            memory_store.resolve_event(event_id, status="resolved")

        for topic in data.get("topics") or []:
            memory_store.touch_topic(topic, quote=user_msg)

    except Exception as e:
        print(f"[extraction] failed applying updates: {e}")


def extract_async(user_msg, assistant_reply):
    """Fire-and-forget — never blocks the conversational turn."""
    thread = threading.Thread(
        target=_run_extraction,
        args=(user_msg, assistant_reply),
        daemon=True,
    )
    thread.start()
