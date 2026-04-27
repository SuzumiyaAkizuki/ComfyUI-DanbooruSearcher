"""
nodes.py
ComfyUI node definitions.

Nodes:
  DanbooruTagSearch   semantic search, outputs a candidate tag pool
  DanbooruRelated     co-occurrence recommendation, expands the pool for LLM filtering
"""

from __future__ import annotations

import os
import sys
import asyncio

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core.engine import DanbooruTagger
from core.models import SearchRequest


def _get_tagger(model_path: str | None) -> DanbooruTagger:
    """Synchronously obtain the engine singleton, compatible with asyncio environments."""
    kwargs = {
        "csv_file":  os.path.join(_HERE, "origin_database", "tags_enhanced.csv"),
        "cooc_file": os.path.join(_HERE, "origin_database", "cooccurrence_clean.parquet"),
        "cache_dir": os.path.join(_HERE, "tags_embedding"),
    }
    if model_path:
        kwargs["model_path"] = model_path

    async def _get():
        return await DanbooruTagger.get_instance(**kwargs)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _get())
                return future.result()
        else:
            return loop.run_until_complete(_get())
    except RuntimeError:
        return asyncio.run(_get())


class DanbooruTagSearch:
    """
    Semantic search node.
    Accepts a natural language description and returns a matched Danbooru tag pool.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "一个在雨中奔跑的少女",
                }),
                "model_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "BGE-M3 local path; leave empty to auto-download from HuggingFace (BAAI/bge-m3)",
                }),
                "top_k": ("INT", {
                    "default": 5, "min": 1, "max": 50, "step": 1,
                    "tooltip": "Top-k candidates retrieved per segment per vector layer",
                }),
                "limit": ("INT", {
                    "default": 80, "min": 10, "max": 300, "step": 10,
                    "tooltip": "Maximum number of tags in the final output",
                }),
                "popularity_weight": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Weight of post-count popularity in the composite score; 0.15 recommended",
                }),
                "use_segmentation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable to segment input and search each concept separately (full scene); disable for exact/single-phrase lookup",
                }),
                "show_nsfw": ("BOOLEAN", {
                    "default": True,
                }),
            }
        }

    RETURN_TYPES = ("DANBOORU_RESULT", "STRING", "STRING")
    RETURN_NAMES = ("search_result", "tags_string", "debug_info")
    FUNCTION = "search"
    CATEGORY = "utils/prompt"

    def search(self, text, model_path, top_k, limit, popularity_weight,
               use_segmentation, show_nsfw):

        if model_path.strip():
            os.environ["DANBOORU_MODEL_PATH"] = model_path.strip()

        tagger = _get_tagger(model_path.strip() or None)

        request = SearchRequest(
            query=text,
            top_k=top_k,
            limit=limit,
            popularity_weight=popularity_weight,
            show_nsfw=show_nsfw,
            use_segmentation=use_segmentation,
        )
        response = tagger.search(request)

        lines = []

        if response.keywords:
            lines.append(f"segmentation keywords: {', '.join(response.keywords)}")
            lines.append("")

        lines += [
            f"{'tag':<28} {'score':<7} {'sem':<7} {'source':<12} cn_name",
            "-" * 80,
        ]
        for r in response.results:
            lines.append(
                f"{r.tag:<28} {r.final_score:<7.3f} {r.semantic_score:<7.3f}"
                f" {str(r.source)[:10]:<12} {r.cn_name}"
            )

        return (
            response,
            response.tags_sfw if not show_nsfw else response.tags_all,
            "\n".join(lines),
        )


class DanbooruRelated:
    """
    Co-occurrence recommendation node (RAG expansion).

    Uses the top-1 tag per segment from the search result as seeds,
    queries the co-occurrence table for related tags, and merges them
    into a combined pool for downstream LLM filtering.

    Typical pipeline:
      DanbooruTagSearch -> search_result -> DanbooruRelated -> combined_tags -> LLM
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "search_result": ("DANBOORU_RESULT",),
                "limit": ("INT", {
                    "default": 50, "min": 10, "max": 200, "step": 10,
                    "tooltip": "Maximum number of co-occurrence recommendations",
                }),
                "show_nsfw": ("BOOLEAN", {
                    "default": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("related_tags", "combined_tags")
    FUNCTION = "related"
    CATEGORY = "utils/prompt"

    def related(self, search_result, limit, show_nsfw):
        tagger = _get_tagger(None)

        # Pick the highest-scoring tag per source segment as the seed.
        seeds: dict[str, object] = {}
        for r in search_result.results:
            src = r.source
            if src not in seeds or r.final_score > seeds[src].final_score:
                seeds[src] = r

        seed_tags = [r.tag for r in seeds.values()]

        # Exclude all tags already present in the search result to avoid duplicates.
        existing = {r.tag for r in search_result.results}

        related = tagger.get_related(
            seed_tags=seed_tags,
            exclude=existing,
            limit=limit,
            show_nsfw=show_nsfw,
        )

        related_str = ", ".join(r.tag for r in related)

        search_tags = (search_result.tags_sfw if not show_nsfw
                       else search_result.tags_all)
        seen = set()
        combined_list = []
        for tag in (search_tags + ", " + related_str).split(", "):
            tag = tag.strip()
            if tag and tag not in seen:
                seen.add(tag)
                combined_list.append(tag)

        return (related_str, ", ".join(combined_list))