# dataset/vision_pair_generator.py
# make image + OCR ground truth pairs
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os, json
from uuid import uuid4

def render_text_image(text, path, w=800, h=400, font_path=None):
    im = Image.new('RGB',(w,h),'white')
    d = ImageDraw.Draw(im)
    if font_path:
        f = ImageFont.truetype(font_path,20)
    else:
        f = None
    d.text((10,10), text, font=f, fill=(0,0,0))
    # optional camera sim
    im = im.filter(ImageFilter.GaussianBlur(radius=random.choice([0,0.5,1.0])))
    im.save(path)

def generate_pairs(infile, outdir):
    os.makedirs(outdir,exist_ok=True)
    with open(infile,'r',encoding='utf8') as f, open(outdir+'/manifest.jsonl','w',encoding='utf8') as mf:
        for l in f:
            it=json.loads(l)
            if it.get('label')==1:
                imgname=f'{str(uuid4())}.png'
                render_text_image(it['text'], os.path.join(outdir,imgname))
                mf.write(json.dumps({'image':imgname,'ocr_text':it['text'],'id':it['id']})+'\\n')
