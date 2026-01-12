def test_metrics_endpoint(client):
    # hit classify a few times
    client.post("/classify", json={"text": "hello"})
    client.post("/classify", json={"text": "My OTP is 123456"})
    client.post("/classify", json={"text": "Call me at 9876543210"})

    r = client.get("/metrics")
    assert r.status_code == 200

    data = r.json()

    # minimal guarantees
    assert "requests" in data
    assert "blocked_credential" in data
    assert "blocked_regex" in data
    assert data["requests"] >= 3
