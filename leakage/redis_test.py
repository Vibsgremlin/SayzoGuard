import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

r.set("sayzoguard_test", "ok")
value = r.get("sayzoguard_test")

print("Redis value:", value)
