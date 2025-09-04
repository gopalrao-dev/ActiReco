# backend/config.py
import os
from dotenv import load_dotenv
from typing import Optional

# Load .env from project root (only once)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env_path = os.path.join(base_dir, ".env")
load_dotenv(env_path)

# App configuration read from environment (with defaults)
ADMIN_API_KEY: Optional[str] = os.getenv("ADMIN_API_KEY")  # keep optional; enforce where needed
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8000"))
DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

# Paths
LOG_DIR: str = os.path.join(base_dir, "logs")
DATA_DIR: str = os.path.join(base_dir, "data")
MODELS_DIR: str = os.path.join(os.path.dirname(__file__), "models")

# Helper to assert admin key exists (call from startup or admin endpoints if you want)
def require_admin_key() -> str:
    if not ADMIN_API_KEY:
        raise RuntimeError("ADMIN_API_KEY is not set in .env")
    return ADMIN_API_KEY