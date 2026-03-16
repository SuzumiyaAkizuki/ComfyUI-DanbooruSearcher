"""
nodes.py
────────
ComfyUI 节点定义。

节点列表：
  DanbooruTagSearch   语义搜索，输出候选标签池
  DanbooruRelated     共现推荐，基于搜索结果扩充候选池，供 LLM 做最终筛选
"""

from __future__ import annotations

import os
import sys
import asyncio

# 把插件目录加入 sys.path，确保 core/ 可以被找到
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core.engine import DanbooruTagger
from core.models import SearchRequest


def _get_tagger(model_path: str | None) -> DanbooruTagger:
    """同步获取引擎单例，兼容 asyncio 环境。"""
    # 数据文件路径全部用绝对路径，避免受 ComfyUI 工作目录影响
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


# ══════════════════════════════════════════════════════════════════════
# 节点 1：语义搜索
# ══════════════════════════════════════════════════════════════════════

class DanbooruTagSearch:
    """
    语义搜索节点。
    输入自然语言描述，返回匹配的 Danbooru 标签候选池。
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
                    "tooltip": "BGE-M3 本地路径，留空则自动从 HuggingFace 下载（BAAI/bge-m3）",
                }),
                "top_k": ("INT", {
                    "default": 5, "min": 1, "max": 50, "step": 1,
                    "tooltip": "每个分词各取前 top_k 个候选标签",
                }),
                "limit": ("INT", {
                    "default": 80, "min": 10, "max": 300, "step": 10,
                    "tooltip": "最终输出的标签数量上限",
                }),
                "popularity_weight": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "热度在综合得分中的权重，推荐 0.15",
                }),
                "use_segmentation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "开启后对输入分词后分别检索，适合完整句子；关闭后整句检索，适合精准查找",
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

        # 设置本地模型路径（engine 会自动处理空字符串）
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

        # debug_info 表格
        lines = [
            f"{'标签':<28} {'综合分':<7} {'语义分':<7} {'来源':<12} 中文含义",
            "-" * 80,
        ]
        for r in response.results:
            lines.append(
                f"{r.tag:<28} {r.final_score:<7.3f} {r.semantic_score:<7.3f}"
                f" {str(r.source)[:10]:<12} {r.cn_name}"
            )

        return (response, response.tags_sfw if not show_nsfw else response.tags_all,
                "\n".join(lines))


# ══════════════════════════════════════════════════════════════════════
# 节点 2：共现推荐
# ══════════════════════════════════════════════════════════════════════

class DanbooruRelated:
    """
    共现推荐节点（RAG 扩充用）。

    以搜索结果中「每个分词的 top1 标签」作为种子，
    从共现表查找关联标签，扩充候选池后一并传给 LLM 做最终筛选。

    推荐接在 DanbooruTagSearch 之后使用：
      DanbooruTagSearch → search_result → DanbooruRelated
                                        → combined_tags → LLM
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "search_result": ("DANBOORU_RESULT",),
                "limit": ("INT", {
                    "default": 50, "min": 10, "max": 200, "step": 10,
                    "tooltip": "共现推荐的标签数量上限",
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

        # ── 每个分词取 final_score 最高的一条作为种子 ──────────────────
        seeds: dict[str, object] = {}
        for r in search_result.results:
            src = r.source
            if src not in seeds or r.final_score > seeds[src].final_score:
                seeds[src] = r

        seed_tags = [r.tag for r in seeds.values()]

        # ── 已选标签集合（种子 + 搜索结果全集，避免推荐重复内容）──────
        existing = {r.tag for r in search_result.results}

        # ── 共现推荐 ──────────────────────────────────────────────────
        related = tagger.get_related(
            seed_tags=seed_tags,
            exclude=existing,
            limit=limit,
            show_nsfw=show_nsfw,
        )

        related_str = ", ".join(r.tag for r in related)

        # combined = 搜索结果 + 推荐，去重，保持顺序
        search_tags = (search_result.tags_sfw if not show_nsfw
                       else search_result.tags_all)
        seen = set()
        combined_list = []
        for tag in (search_tags + ", " + related_str).split(", "):
            tag = tag.strip()
            if tag and tag not in seen:
                seen.add(tag)
                combined_list.append(tag)

        combined_str = ", ".join(combined_list)

        return (related_str, combined_str)