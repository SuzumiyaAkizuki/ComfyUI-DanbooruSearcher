from .nodes import DanbooruTagSearch, DanbooruRelated

NODE_CLASS_MAPPINGS = {
    "DanbooruTagSearch": DanbooruTagSearch,
    "DanbooruRelated":   DanbooruRelated,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DanbooruTagSearch": "Danbooru Smart Search",
    "DanbooruRelated":   "Danbooru Related Tags",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
