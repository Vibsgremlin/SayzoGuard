import requests

URL = "http://127.0.0.1:8000/classify"

def post(text):
    return requests.post(URL, json={"text": text}).json()

def test_low_signal_safe():
    r = post("Project update looks fine")
    assert r["blocked"] is False
    assert r["reason"] in ("low_signal_safe", "safe_content")

def test_price_negotiation_allowed():
    r = post("Budget is 5000, can we negotiate?")
    assert r["blocked"] is False

def test_phone_blocked():
    r = post("Call me at 9876543210")
    assert r["blocked"] is True

def test_otp_blocked():
    r = post("My OTP is 123456")
    assert r["blocked"] is True

def test_low_signal_is_safe():
    r = post("You can reach me later")
    assert r["blocked"] is False
    assert r["reason"] == "low_signal_safe"

