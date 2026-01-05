# dataset/gen_synthetic.py
# Synthetic dataset generator for contact-leakage prevention model

import os
import random
import json
from faker import Faker

fake = Faker()

SAFE_PATTERNS = [
    "schedule a meeting",
    "share the document",
    "project update",
    "team sync",
    "deployment plan",
]

LEAK_PATTERNS = [
    "my password is {pwd}",
    "login using my otp {otp}",
    "here is my aadhaar {aadhaar}",
    "bank account number {acc}",
    "gmeet link: {link}",
]


def random_pwd():
    return fake.password(length=10)


def random_otp():
    return str(random.randint(100000, 999999))


def random_acc():
    return str(random.randint(1000000000, 9999999999))


def random_aadhaar():
    return "".join(str(random.randint(0, 9)) for _ in range(12))


def random_gmeet():
    return f"https://meet.google.com/{fake.lexify('???')}-{fake.lexify('???')}-{random.randint(100, 999)}"


def generate_sample(leak=False):
    if not leak:
        text = random.choice(SAFE_PATTERNS)
        label = 0
    else:
        pattern = random.choice(LEAK_PATTERNS)
        text = pattern.format(
            pwd=random_pwd(), otp=random_otp(), aadhaar=random_aadhaar(), acc=random_acc(), link=random_gmeet()
        )
        label = 1

    return {"text": text, "label": label}


def generate_dataset(n_safe=5000, n_leak=5000, out_path="synthetic_dataset.jsonl"):
    data = []

    for _ in range(n_safe):
        data.append(generate_sample(leak=False))

    for _ in range(n_leak):
        data.append(generate_sample(leak=True))

    random.shuffle(data)

    with open(out_path, "w", encoding="utf8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset saved to {out_path} with {len(data)} samples.")


if __name__ == "__main__":
    generate_dataset()
