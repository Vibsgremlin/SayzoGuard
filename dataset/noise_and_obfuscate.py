# dataset/noise_and_obfuscate.py - adversarial obfuscations


# generates spaced numbers, number words, leetspeak, homoglyphs, common delimiters
# and split tokens, small encodings, base64-like encodings, zero-width char insertions,
# and simple substituion maps.

import re, json, random
from uuid import uuid4

NUM2WORD = {'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine'}

def spaced_digits(s): return ' '.join(list(s))
def word_numbers(s): return ' '.join(NUM2WORD.get(c,c) for c in s)
def leetspeak(s):
    map_ = {'a':'@','e':'3','i':'1','o':'0','s':'$','t':'7'}
    return ''.join(map_.get(c.lower(),c) for c in s)
def homoglyphs(s):
    m={'0':'O','1':'l','2':'Z','5':'S','8':'B'}
    return ''.join(m.get(c,c) for c in s)
def insert_zero_width(s):
    zw = '\u200b'
    return zw.join(list(s))

OBF_METHODS = [spaced_digits, word_numbers, leetspeak, homoglyphs, insert_zero_width]

def obfuscate_numeric_run(match):
    s = match.group(0)
    return random.choice(OBF_METHODS)(s)

def obfuscate_text(text, n_variants=4):
    variants=[]
    for _ in range(n_variants):
        v = re.sub(r'\\d{3,}', obfuscate_numeric_run, text)
        variants.append(v)
    return variants

# example runner reading paraphrased_dataset.jsonl and writing adversarial_dataset.jsonl
def expand_file(inpath, outpath):
    out=[]
    with open(inpath,'r',encoding='utf8') as f:
        for l in f:
            it=json.loads(l)
            if it.get('label')==1:
                for v in obfuscate_text(it['text'], n_variants=6):
                    new=dict(it); new['text']=v; new['id']=str(uuid4()); new['layer']='adversarial'
                    out.append(new)
    with open(outpath,'w',encoding='utf8') as o:
        for it in out: o.write(json.dumps(it)+'\\n')
    print('wrote', len(out))
