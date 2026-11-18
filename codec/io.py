"""Versioned, explicit metadata and byte accounting (no training state)."""
import json
from pathlib import Path

FORMAT = 'hpc-v1'

def json_value(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2) + '\n')

def read_json(path):
    return json.loads(Path(path).read_text())
