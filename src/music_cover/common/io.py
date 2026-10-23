from __future__ import annotations
import hashlib
import json
from pathlib import Path

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

def identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def artifact_paths(directory):
    """Include every payload in stage cache validation, including nested files."""
    directory = Path(directory)
    return {str(p.relative_to(directory)): p for p in sorted(directory.rglob("*"))
            if p.is_file() and p.stat().st_size and p.name not in
            {"process.log", "command.json", "result.json", "failure.json"}}
