#REGEX FIREWALL
#detects phone numbers, emails, URLs, and forbidden links.


import re

FORBIDDEN_LINK_DOMAINS = [
    "meet.google.com","zoom.us","wa.me","web.whatsapp.com",
    "instagram.com","facebook.com","linkedin.com","t.me",
    "telegram.me","snapchat.com","anydesk.com"
]

def contains_forbidden_link(text: str) -> bool:
    t = text.lower()
    for d in FORBIDDEN_LINK_DOMAINS:
        if d in t:
            return True
    return False

def basic_rule_score(text: str) -> int:
    score = 0
    if re.search(r"\b\d{10}\b", text):
        score += 3
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        score += 2
    if re.search(r'https?://\S+', text):
        score += 2
    for d in FORBIDDEN_LINK_DOMAINS:
        if d in text.lower():
            score += 3
    return score
