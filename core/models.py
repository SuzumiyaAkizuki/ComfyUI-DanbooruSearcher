from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TagResult:
    tag: str
    cn_name: str
    category: str
    nsfw: str
    final_score: float
    semantic_score: float
    count: int
    source: str
    layer: str
    wiki: str = ""


@dataclass
class RelatedTag:
    tag: str
    cn_name: str
    category: str
    nsfw: str
    cooc_count: int
    cooc_score: float
    sources: list[str] = field(default_factory=list)
    post_count: int = 0   # 该标签的发帖总数（与 search_tags 的 count 对齐）
    wiki: str = ""        # 标签 wiki 描述


@dataclass
class SearchRequest:
    query: str
    top_k: int = 5
    limit: int = 80
    popularity_weight: float = 0.15
    show_nsfw: bool = True
    use_segmentation: bool = True
    target_layers: list[str] = field(
        default_factory=lambda: ['英文', '中文扩展词', '释义', '中文核心词']
    )
    target_categories: list[str] = field(
        default_factory=lambda: ['General', 'Character', 'Copyright']
    )


@dataclass
class SearchResponse:
    tags_all: str
    tags_sfw: str
    results: list[TagResult]
    keywords: list[str]
    segments: list[str] = field(default_factory=list)  # 分隔符切分后的原始从句级片段
