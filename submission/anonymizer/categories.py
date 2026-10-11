import json
from dataclasses import dataclass
from pathlib import Path

CATEGORIES_PATH = Path(__file__).with_name("categories.json")


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    description: str
    examples: list[str]
    do_not_mask: list[str]


def load_categories(path: Path = CATEGORIES_PATH) -> list[Category]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    return [Category(**entry) for entry in entries]
