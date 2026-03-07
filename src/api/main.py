from __future__ import annotations

import uvicorn

from src.api.app import app  

def main() -> None:
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()