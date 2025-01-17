task_set = [
    {
        "name": "diamond_pickaxe",
        "description": "craft a diamond pickaxe",
        "group": ["craft_item", "diamond", "pickaxe", "tool"],
        "evaluation": {
            "stats": {"craft_item:diamond_pickaxe": 1}
            },
        "reward": 1,
        "max_steps": 25000
    },
    {
        "name": "iron_axe",
        "description": "craft an iron axe",
        "group": ["craft_item", "iron", "axe", "tool"],
        "evaluation": {
            "stats": {"craft_item:iron_axe": 1}
            },
        "reward": 1,
        "max_steps": 15000
    },
    {
        "name": "diamond_ore",
        "description": "mine a diamond ore",
        "group": ["mine_block", "diamond", "ore"],
        "evaluation": {
            "stats": {"mine_block:diamond_ore": 1}
            },
        "reward": 1,
        "max_steps": 25000
    }
]
