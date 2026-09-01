import json
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(self, file_path: str = "memory.json"):
        self.file_path = Path(file_path)

    def load(self) -> dict[str, Any]:
        if not self.file_path.exists():
            return {}

        with self.file_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            content = file.read()

            if not content.strip():
                return {}

            return json.loads(content)


    def save(self, memory: dict[str, Any]) -> None:
        with self.file_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                memory,
                file,
                ensure_ascii=False,
                indent=2
            )