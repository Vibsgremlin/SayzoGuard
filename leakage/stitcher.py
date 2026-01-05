#STITCHER(Multi-turn Conversations)

from collections import deque

SESSIONS = {}

def add_message(session_id, text):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = deque(maxlen=5)
    SESSIONS[session_id].append(text)

def get_stitched(session_id):
    if session_id not in SESSIONS:
        return ""
    return "\n".join(SESSIONS[session_id])
