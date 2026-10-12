from pathlib import Path


def read_key(path) -> str:
    raw = Path(path).read_text(encoding="utf-8").strip()
    if "=" in raw and "\n" not in raw and raw.split("=", 1)[0].replace("export ", "").strip().isidentifier():
        raw = raw.split("=", 1)[1].strip()
    key = raw.strip("\"'")
    if not key or any(c.isspace() for c in key):
        raise ValueError("key file must contain one API key, optionally as KEY=value")
    return key

