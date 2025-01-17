import json
from pathlib import Path
from typing import (
    Optional, Sequence, List, Tuple, Dict, Union, Callable
)

FILE_DIR = Path(__file__).parent

RECIPES_DIR = FILE_DIR / "recipes"
PLANS_DIR = FILE_DIR / "plans"
TASKS_DIR = FILE_DIR / "tasks"

TAG_ITEMS_FILE = FILE_DIR / "tag_items.json"
