# dataset/conversation_simulator.py
# multi turn leakage threads
# simulate threads where leakage can be split across turns or appear after context.

import json, random
from uuid import uuid4

BASE_TEMPLATES = [
    ["Hey can you help with my bank transfer","Sure what details","Account number is 9876543210","Thanks done"],
    ["Need to share meeting link","Ok","https://meet.google.com/xxx-yyyy-zzz","Join now"]
]

def make_thread(template):
    thread=[]
    for i,txt in enumerate(template):
        thread.append({'turn':i,'speaker':'user' if i%2==0 else 'agent','text':txt,'id':str(uuid4())})
    label = any(re.search(r'(\\d{4,}|meet\\.google\\.com|otp)',t['text'],re.I) for t in thread)
    return {'thread':thread,'label':int(label)}

def generate_threads(n, out_path):
    with open(out_path,'w',encoding='utf8') as f:
        for _ in range(n):
            tpl = random.choice(BASE_TEMPLATES)
            f.write(json.dumps(make_thread(tpl))+'\\n')
