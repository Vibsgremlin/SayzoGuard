# dataset/generate_paraphrases.py
import json, csv, os, time
from uuid import uuid4

# TODO replace with real LLM wrapper call
def call_llm_paraphrase(prompt, n=5):
    # example placeholder — return n fake paraphrases
    return [prompt + f' (paraphrase {i})' for i in range(n)]

def augment_with_paraphrases(input_path, out_jsonl, out_review_csv, per_item=5):
    os.makedirs(os.path.dirname(out_jsonl) or '.', exist_ok=True)
    write_csv_rows = []
    out_items = []
    with open(input_path,'r',encoding='utf8') as f:
        for line in f:
            item = json.loads(line)
            if item.get('label')==0:
                continue
            base = item['text']
            paraphrases = call_llm_paraphrase(base, per_item)
            for p in paraphrases:
                new_item = dict(item)
                new_item['text'] = p
                new_item['id'] = str(uuid4())
                new_item['layer'] = 'semi_synthetic_paraphrase'
                out_items.append(new_item)
                write_csv_rows.append({'id':new_item['id'],'original':base,'paraphrase':p,'label':''})
    with open(out_jsonl,'w',encoding='utf8') as outf:
        for it in out_items:
            outf.write(json.dumps(it)+'\n')
    # create CSV for quick human review/labeling
    with open(out_review_csv,'w',encoding='utf8') as csvf:
        w = csv.DictWriter(csvf,fieldnames=['id','original','paraphrase','label'])
        w.writeheader()
        w.writerows(write_csv_rows)
    print('wrote', len(out_items), 'paraphrases and review CSV')
