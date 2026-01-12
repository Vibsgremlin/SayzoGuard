from slowapi import Limiter
from slowapi.util import get_remote_address
from redis import Redis

redis = Redis(host="localhost", port=6379, decode_responses=True)

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379"
)
