#NORMALIZER
#handles spelled digits, misspellings, emoji digits, and removes formatting.


import re

DIGIT_WORDS = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4",
    "five":"5","six":"6","seven":"7","eight":"8","nine":"9",
    "oewn":"1","tewo":"2","tree":"3","phour":"4",
    "fyve":"5","sevvn":"7","nien":"9","ate":"8"
}

EMOJI_MAP = {"0️⃣":"0","1️⃣":"1","2️⃣":"2","3️⃣":"3","4️⃣":"4","5️⃣":"5","6️⃣":"6","7️⃣":"7","8️⃣":"8","9️⃣":"9"}

def normalize(text: str) -> str:
    text = text.lower()

    for e,d in EMOJI_MAP.items():
        text = text.replace(e, d)

    for word, digit in DIGIT_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)

    text = re.sub(r"[\s\-\.]", "", text)
    return text
