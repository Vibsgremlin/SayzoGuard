# NORMALIZER
# Handles:
# - spelled digits
# - misspellings
# - emoji digits
# - unicode tricks
# - formatting removal
# - keeps price semantics intact

import re
import unicodedata

DIGIT_WORDS = {
    # standard
    "zero":"0","one":"1","two":"2","three":"3","four":"4",
    "five":"5","six":"6","seven":"7","eight":"8","nine":"9",

    # common misspellings / obfuscations
    "oewn":"1","tewo":"2","tree":"3","phour":"4",
    "fyve":"5","sevvn":"7","nien":"9","ate":"8",
}

EMOJI_MAP = {
    "0️⃣":"0","1️⃣":"1","2️⃣":"2","3️⃣":"3","4️⃣":"4",
    "5️⃣":"5","6️⃣":"6","7️⃣":"7","8️⃣":"8","9️⃣":"9"
}

# characters often used to hide digits
JUNK_CHARS = r"[\u200b-\u200f\u202a-\u202e]"


def normalize(text: str) -> str:
    """
    Normalizes text to remove evasion tricks while
    preserving semantic meaning for price negotiation.
    """

    # 1. Unicode normalization (kills fancy unicode tricks)
    text = unicodedata.normalize("NFKC", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Remove zero-width / invisible chars
    text = re.sub(JUNK_CHARS, "", text)

    # 4. Convert emoji digits
    for e, d in EMOJI_MAP.items():
        text = text.replace(e, d)

    # 5. Convert spelled / misspelled digits
    for word, digit in DIGIT_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)

    # 6. Collapse spaced or punctuated digits: "9 - 7 . 1 2"
    text = re.sub(r"(?<=\d)[\s\-\.]+(?=\d)", "", text)

    # 7. Normalize excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
