from .search import DanbooruTagSearch

NODE_CLASS_MAPPINGS = {
    "DanbooruTagSearch": DanbooruTagSearch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DanbooruTagSearch": "Danbooru Smart Search"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']