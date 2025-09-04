# run.py
import uvicorn
import sys
from backend import config

if __name__ == "__main__":
    try:
        print(f"🚀 Starting ActiReco at http://{config.HOST}:{config.PORT} (debug={config.DEBUG})")
        uvicorn.run(
            "backend.app:app",
            host=config.HOST,
            port=config.PORT,
            reload=config.DEBUG,   # auto-reload only if DEBUG=true
            log_level="debug" if config.DEBUG else "info",
        )
    except Exception as e:
        print(f"❌ Failed to start ActiReco: {e}", file=sys.stderr)
        sys.exit(1)