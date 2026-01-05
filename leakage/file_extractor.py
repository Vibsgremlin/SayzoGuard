#OCR+FILE TEXT EXTRACTION


from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from io import BytesIO

def extract_text_from_file(uploaded_file):

    file_type = uploaded_file.type

    if file_type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    if file_type.startswith("image/"):
        img = Image.open(BytesIO(uploaded_file.read()))
        return pytesseract.image_to_string(img)

    content = uploaded_file.read()
    return content.decode(errors="ignore")
