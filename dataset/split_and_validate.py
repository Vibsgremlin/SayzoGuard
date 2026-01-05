# dataset/split_and_validate.py
# combine, dedupe, stratify and split

import json, random
from sklearn.model_selection import train_test_split

def load_files(paths):
    items=[]
    seen=set()
    for p in paths:
        with open(p,'r',encoding='utf8') as f:
            for l in f:
                it=json.loads(l)
                if it.get('id') in seen: continue
                seen.add(it.get('id'))
                items.append(it)
    return items

def stratified_split(items, out_prefix='sayzoguard'):
    labels=[it['label'] for it in items]
    train, test = train_test_split(items, test_size=0.2, stratify=labels, random_state=42)
    train, val = train_test_split(train, test_size=0.125, stratify=[t['label'] for t in train], random_state=42)
    for name, col in [('train',train),('val',val),('test',test)]:
        with open(f'{out_prefix}_{name}.jsonl','w',encoding='utf8') as f:
            for it in col: f.write(json.dumps(it)+'\\n')
    print('done splits', len(train), len(val), len(test))
