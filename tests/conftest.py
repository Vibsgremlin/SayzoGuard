import sys
import os
import pytest
from fastapi.testclient import TestClient

# --- FIX PYTHON PATH (2 levels up) ---
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sayzoguard.inference_server.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)
