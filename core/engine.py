"""
core/engine.py
DanbooruTagger core engine for the ComfyUI plugin.

Cache layout (under cache_dir/):
  danbooru_multiview_embeddings.safetensors  — four-view FP16 tensor matrix
  tags_metadata.parquet                      — DataFrame (name/cn_name/cn_core/wiki/nsfw/category/post_count)
  version_data.json                          — scalar metadata (schema_version, updated_at)
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import jieba
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file as st_load, save_file as st_save
from sentence_transformers import SentenceTransformer

from .models import SearchRequest, SearchResponse, TagResult


# ──────────────────────────────────────────────
# 停用词
# ──────────────────────────────────────────────

STOP_WORDS: frozenset[str] = frozenset({
    ',', '.', ':', ';', '?', '!', '"', "'", '`',
    '(', ')', '[', ']', '{', '}', '<', '>',
    '-', '_', '=', '+', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '~',
    '，', '。', '：', '；', '？', '！', '“', '”', '‘', '’',
    '（', '）', '【', '】', '《', '》', '、', '…', '—', '·',
    ' ', '\t', '\n', '\r',
    '的', '地', '得', '了', '着', '过',
    '是', '为', '被', '给', '把', '让', '由',
    '在', '从', '自', '向', '往', '对', '于',
    '和', '与', '及', '或', '且', '而', '但', '并', '即', '又', '也',
    '啊', '吗', '吧', '呢', '噢', '哦', '哈', '呀', '哇',
    '我', '你', '他', '她', '它', '我们', '你们', '他们',
    '这', '那', '此', '其', '谁', '啥', '某', '每',
    '这个', '那个', '这些', '那些', '这里', '那里',
    '个', '位', '只', '条', '张', '幅', '件', '套', '双', '对', '副',
    '种', '类', '群', '些', '点', '份', '部', '名',
    '很', '太', '更', '最', '挺', '特', '好', '真',
    '一', '一个', '一种', '一下', '一点', '一些',
    '有', '无', '非', '没', '不',
    '正在', '已经', '正', '刚', '开始', '继续', '一直', '不断',
    '穿着', '戴着', '穿', '戴',
    '带有', '具有', '拥有',
    '看起来', '看上去', '显得', '仿佛', '似乎',
    '十分', '非常', '特别', '比较',
    '图片', '画面', '图像',
    '位于', '处于',
    '许多', '大量', '各种', '所有', '其他', '其它',
    # ── 英文停用词 ──
    'a', 'an', 'the',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'into',
    'about', 'between', 'through', 'after', 'before', 'above', 'below',
    'and', 'or', 'but', 'nor', 'so', 'yet',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'do', 'does', 'did', 'done',
    'have', 'has', 'had', 'having',
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
    'not', 'no', 'very', 'too', 'also', 'just', 'only', 'even', 'still',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'it', 'its', 'they', 'them', 'their',
    'this', 'that', 'these', 'those', 'which', 'who', 'whom', 'what',
    'there', 'here', 'where', 'when', 'how', 'all', 'each', 'every',
    'some', 'any', 'few', 'more', 'most', 'other', 'such',
    'than', 'up', 'out', 'if', 'then', 'else', 'while', 'during',
    'both', 'same', 'own', 'now',
})

CAT_MAP: dict[str, str] = {
    '0': 'General', '1': 'Artist', '3': 'Copyright', '4': 'Character', '5': 'Meta',
}

LOCAL_MODEL_PATH = 'my_model_bge_m3'
HF_MODEL_ID      = 'BAAI/bge-m3'
SCHEMA_VERSION   = 3

# 用户显式分隔后，纯 CJK 片段超过此长度仍用 jieba 切分
_ATOMIC_CJK_MAX_LEN = 7

# 四路 embedding 层配置: (层名, tensor 属性名, DataFrame 列名)
_LAYER_SPEC: list[tuple[str, str, str]] = [
    ('英文',   'emb_en',      'name'),
    ('中文扩展词', 'emb_cn',      'cn_name'),
    ('释义',   'emb_wiki',    'wiki'),
    ('中文核心词', 'emb_cn_core', 'cn_core'),
]
_ALL_LAYER_NAMES = [ln for ln, _, _ in _LAYER_SPEC]

# 英文复合标签最大单词数
_EN_MAX_COMPOUND = 4


# ──────────────────────────────────────────────
# LRU 缓存
# ──────────────────────────────────────────────

class LRUCache:
    def __init__(self, maxsize: int):
        self._cache: OrderedDict[Any, Any] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: Any) -> Any:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: Any, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = value

    def __len__(self) -> int:
        return len(self._cache)


# ──────────────────────────────────────────────
# 缓存路径助手
# ──────────────────────────────────────────────

class _CachePaths:
    def __init__(self, cache_dir: str | Path):
        self.dir        = Path(cache_dir)
        self.embeddings = self.dir / 'danbooru_multiview_embeddings.safetensors'
        self.metadata   = self.dir / 'tags_metadata.parquet'
        self.meta_json  = self.dir / 'version_data.json'

    def exists(self) -> bool:
        return (
            self.embeddings.is_file()
            and self.metadata.is_file()
            and self.meta_json.is_file()
        )

    def ensure_dir(self):
        self.dir.mkdir(parents=True, exist_ok=True)


class DanbooruTagger:
    """Core search engine (singleton)."""

    _instance: Optional['DanbooruTagger'] = None
    _lock: Optional[asyncio.Lock] = None

    @classmethod
    def is_ready(cls) -> bool:
        return cls._instance is not None and cls._instance.is_loaded

    @classmethod
    async def get_instance(cls, **kwargs) -> 'DanbooruTagger':
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        async with cls._lock:
            if cls._instance is None:
                inst = cls(**kwargs)
                await asyncio.to_thread(inst.load)
                cls._instance = inst
            return cls._instance

    def __init__(
        self,
        model_path: Optional[str] = None,
        csv_file:   str = 'origin_database/tags_enhanced.csv',
        cache_dir:  str = 'tags_embedding',
        cooc_file:  str = 'origin_database/cooccurrence_clean.parquet',
    ):
        if model_path:
            self.model_path = model_path
        elif os.path.exists(LOCAL_MODEL_PATH):
            print(f'[Engine] local model found at "{LOCAL_MODEL_PATH}"')
            self.model_path = LOCAL_MODEL_PATH
        else:
            print(f'[Engine] local model not found, will use HF id "{HF_MODEL_ID}"')
            self.model_path = HF_MODEL_ID

        self.csv_path  = csv_file
        self.device    = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.paths     = _CachePaths(cache_dir)
        self.cooc_file = cooc_file

        self.model:         Optional[SentenceTransformer]       = None
        self.df:            Optional[pd.DataFrame]              = None
        self.emb_en:        Optional[torch.Tensor]              = None
        self.emb_cn:        Optional[torch.Tensor]              = None
        self.emb_wiki:      Optional[torch.Tensor]              = None
        self.emb_cn_core:   Optional[torch.Tensor]              = None
        self.max_log_count: float                               = 15.0
        self.cooc:          dict[str, list[tuple[str, int]]]    = {}
        self._name_to_idx:  dict[str, int]                      = {}
        self.is_loaded:     bool                                = False

        # 预提取的列数组，避免热点路径上反复执行 df.iloc[idx]
        self._arr_name:       Optional[np.ndarray] = None
        self._arr_cn_name:    Optional[np.ndarray] = None
        self._arr_category:   Optional[np.ndarray] = None
        self._arr_nsfw:       Optional[np.ndarray] = None
        self._arr_wiki:       Optional[np.ndarray] = None
        self._arr_post_count: Optional[np.ndarray] = None
        self._arr_pop_score:  Optional[np.ndarray] = None

        # 三层 LRU 缓存（纯内存，重启后自动重热）
        self._emb_cache:     LRUCache = LRUCache(maxsize=10_000)
        self._search_cache:  LRUCache = LRUCache(maxsize=5_000)
        self._related_cache: LRUCache = LRUCache(maxsize=2_000)

    def load(self) -> None:
        """Synchronous load; call inside a thread pool."""
        if self.is_loaded:
            return
        t0 = time.time()

        if not self.paths.exists():
            print('[Engine] no cache found, building from scratch (may take 1-3 min)...')
            self._load_model()
            self._build_full()
        else:
            print(f'[Engine] loading cache from {self.paths.dir}')
            self._load_from_cache()
            if self._cached_schema_version() != SCHEMA_VERSION:
                print('[Engine] cache schema mismatch, rebuilding...')
                self._load_model()
                self._build_full()
            elif os.path.exists(self.csv_path):
                self._load_model()
                self._smart_update()

        if self.model is None:
            self._load_model()

        self._setup_jieba_from_memory()
        self._load_cooc()
        self._name_to_idx = {n: i for i, n in enumerate(self.df['name'])}
        self._tag_names_set: set[str] = set(self._name_to_idx.keys())
        self._rebuild_arrays_from_df()
        self._normalize_embeddings()
        self.is_loaded = True
        print(f'[Engine] ready in {time.time() - t0:.2f}s')

    def _normalize_embeddings(self) -> None:
        """L2 归一化四路 embedding，使 search 阶段可直接用矩阵乘得 cosine similarity。"""
        for _, attr, _ in _LAYER_SPEC:
            t = getattr(self, attr)
            if t is None:
                continue
            setattr(self, attr, torch.nn.functional.normalize(t, p=2, dim=1))

    def _rebuild_arrays_from_df(self) -> None:
        """将 DataFrame 中搜索热点路径需要的列预提取为 numpy 数组。"""
        if self.df is None:
            return
        self._arr_name       = self.df['name'].to_numpy()
        self._arr_cn_name    = self.df['cn_name'].to_numpy()
        self._arr_category   = self.df['category'].astype(str).to_numpy()
        self._arr_nsfw       = self.df['nsfw'].astype(str).to_numpy()
        self._arr_wiki       = self.df['wiki'].astype(str).to_numpy()
        self._arr_post_count = self.df['post_count'].to_numpy()
        max_log = self.max_log_count if self.max_log_count > 0 else 1.0
        self._arr_pop_score  = np.log1p(self._arr_post_count) / max_log

    # ── 搜索 ──────────────────────────────────────────────────────────────

    def _encode_queries(self, queries: list[str]) -> torch.Tensor:
        """批量编码查询词，命中 embedding 缓存的跳过 model.encode。"""
        cached_vecs: list[Optional[torch.Tensor]] = [self._emb_cache.get(q) for q in queries]
        uncached_idx = [i for i, v in enumerate(cached_vecs) if v is None]

        if uncached_idx:
            uncached_texts = [queries[i] for i in uncached_idx]
            new_embs = self.model.encode(
                uncached_texts, convert_to_tensor=True, show_progress_bar=False,
            ).float()
            new_embs = torch.nn.functional.normalize(new_embs, p=2, dim=1)
            for j, i in enumerate(uncached_idx):
                emb = new_embs[j]
                self._emb_cache.put(queries[i], emb)
                cached_vecs[i] = emb

        return torch.stack(cached_vecs)  # type: ignore[arg-type]

    def search(self, request: SearchRequest) -> SearchResponse:
        if not self.is_loaded:
            self.load()

        cache_key = (
            request.query,
            request.top_k,
            request.limit,
            request.popularity_weight,
            request.use_segmentation,
            tuple(sorted(request.target_layers)),
            tuple(sorted(request.target_categories)),
        )
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return cached

        if request.use_segmentation:
            raw_kw, raw_segments = self._smart_split(request.query)
            keywords = [w.strip() for w in raw_kw if w.strip() and w.strip() not in STOP_WORDS]
            keywords_set = set(keywords)
            extra_segments = [s for s in raw_segments if s != request.query and s not in keywords_set]
            queries = [request.query] + extra_segments + keywords
        else:
            keywords = []
            extra_segments = []
            queries  = [request.query]

        q_emb = self._encode_queries(queries)

        tl    = request.target_layers
        k     = request.top_k

        # 每个查询词单独做意图识别
        query_weights = [self._detect_intent(q) for q in queries]
        active_layers = [ln for ln in _ALL_LAYER_NAMES if ln in tl]

        # 预算每个 query × 每个 layer 的 top_k 配额
        cur_pvk_per_q: list[dict[str, int]] = []
        for cur_weights in query_weights:
            if active_layers:
                aw       = {l: cur_weights.get(l, 1.0) for l in active_layers}
                total_aw = sum(aw.values())
                cur_pvk  = {l: max(1, round(k * aw[l] / total_aw)) for l in active_layers}
            else:
                cur_pvk = {}
            cur_pvk_per_q.append(cur_pvk)

        target_cats = request.target_categories
        w_pop       = request.popularity_weight

        final: dict[str, TagResult] = {}

        # 按 layer 批量做矩阵乘 + topk，合并为 L=4 次大调用
        for ln, attr, _ in _LAYER_SPEC:
            if ln not in tl:
                continue
            emb_matrix = getattr(self, attr)   # (N, D)，已归一化
            if emb_matrix is None:
                continue

            k_max = max((cur_pvk_per_q[i].get(ln, 1) for i in range(len(queries))), default=1)
            k_max = min(k_max, emb_matrix.shape[0])

            scores = q_emb @ emb_matrix.T                       # (Q, N)
            top_v, top_i = scores.topk(k_max, dim=1)            # (Q, k_max)
            top_v_list = top_v.tolist()
            top_i_list = top_i.tolist()

            for i, source_word in enumerate(queries):
                cur_weights = query_weights[i]
                kq = cur_pvk_per_q[i].get(ln, 1)
                layer_w = cur_weights.get(ln, 1.0)
                row_v = top_v_list[i]
                row_i = top_i_list[i]
                for j in range(min(kq, len(row_v))):
                    score = row_v[j]
                    if score < 0.35:
                        break
                    idx = row_i[j]
                    cat_text = CAT_MAP.get(self._arr_category[idx], 'Other')
                    if cat_text not in target_cats:
                        continue
                    tag_name    = self._arr_name[idx]
                    count       = self._arr_post_count[idx]
                    pop_score   = self._arr_pop_score[idx]
                    final_score = score * layer_w * (1 - w_pop) + pop_score * w_pop
                    if tag_name not in final or final_score > final[tag_name].final_score:
                        final[tag_name] = TagResult(
                            tag=tag_name, cn_name=self._arr_cn_name[idx], category=cat_text,
                            nsfw=self._arr_nsfw[idx],
                            final_score=round(float(final_score), 4),
                            semantic_score=round(float(score), 4),
                            count=int(count), source=source_word, layer=ln,
                            wiki=self._arr_wiki[idx],
                        )

        # ── 全句语义一致性软重排 ──────────────────────────────────────────
        full_q = q_emb[0]          # queries[0] 始终为完整原始查询
        alpha  = 0.3 if request.use_segmentation else 0
        for r in final.values():
            idx = self._name_to_idx[r.tag]
            max_co = 0.0
            for ln, attr, _ in _LAYER_SPEC:
                if ln not in tl:
                    continue
                max_co = max(max_co, float(torch.dot(full_q, getattr(self, attr)[idx])))
            r.final_score = round(r.final_score * (1.0 - alpha + alpha * max_co), 4)

        # 收集每个查询源的 top-1 结果（高于阈值）
        guaranteed_tags: set[str] = set()
        for source_word in queries:
            best: TagResult | None = None
            for r in final.values():
                if r.source == source_word and r.final_score > 0.45:
                    if best is None or r.final_score > best.final_score:
                        best = r
            if best is not None:
                guaranteed_tags.add(best.tag)

        sorted_results = sorted(final.values(), key=lambda r: r.final_score, reverse=True)
        valid: list[TagResult] = []
        for r in sorted_results:
            if r.final_score <= 0.45:
                continue
            if len(valid) < request.limit or r.tag in guaranteed_tags:
                valid.append(r)

        tags_all = ', '.join(r.tag for r in valid)
        tags_sfw = ', '.join(r.tag for r in valid if r.nsfw != '1')
        response = SearchResponse(
            tags_all=tags_all, tags_sfw=tags_sfw,
            results=valid, keywords=keywords, segments=extra_segments,
        )
        self._search_cache.put(cache_key, response)
        return response

    def _build_full(self) -> None:
        print(f'[Engine] reading {self.csv_path}')
        raw_df             = self._read_csv_robust(self.csv_path)
        self.df            = self._preprocess_raw_df(raw_df)
        self.max_log_count = float(np.log1p(self.df['post_count'].max()))
        self._encode_all_and_save()

    def _encode_all_and_save(self) -> None:
        print('[Engine] encoding all rows...')
        for _, attr, col in _LAYER_SPEC:
            setattr(self, attr, self._encode_texts(self.df[col].tolist()))
        self._save_cache()

    def _smart_update(self) -> None:
        print('[Engine] checking for incremental changes...')
        t0 = time.time()

        raw_df = self._read_csv_robust(self.csv_path)
        new_df = self._preprocess_raw_df(raw_df)

        _SIG_COLS = ['cn_name', 'wiki', 'cn_core']

        def _sig(df: pd.DataFrame, iloc_idx: int) -> tuple:
            row = df.iloc[iloc_idx]
            return tuple(str(row.get(c, '')) for c in _SIG_COLS)

        cached_idx: dict[str, int] = {n: i for i, n in enumerate(self.df['name'])}
        new_idx:    dict[str, int] = {n: i for i, n in enumerate(new_df['name'])}

        added_names   = [n for n in new_idx if n not in cached_idx]
        deleted_names = [n for n in cached_idx if n not in new_idx]
        changed_names = [
            n for n in new_idx
            if n in cached_idx and _sig(new_df, new_idx[n]) != _sig(self.df, cached_idx[n])
        ]

        if not added_names and not deleted_names and not changed_names:
            print('[Engine] data is up to date, skipping update')
            return

        print(f'[Engine] changes: +{len(added_names)} ~{len(changed_names)} -{len(deleted_names)}')

        if deleted_names:
            keep_mask = ~self.df['name'].isin(set(deleted_names))
            keep_pos  = [i for i, v in enumerate(keep_mask) if v]
            self.df = self.df[keep_mask].reset_index(drop=True)
            for _, attr, _ in _LAYER_SPEC:
                setattr(self, attr, getattr(self, attr)[keep_pos])
            cached_idx = {n: i for i, n in enumerate(self.df['name'])}

        if changed_names:
            changed_rows = new_df[new_df['name'].isin(set(changed_names))].reset_index(drop=True)
            _vecs = {attr: self._encode_texts(changed_rows[col].tolist()) for _, attr, col in _LAYER_SPEC}
            for j, name in enumerate(changed_rows['name']):
                ci = cached_idx[name]
                for _, attr, _ in _LAYER_SPEC:
                    getattr(self, attr)[ci] = _vecs[attr][j]
                for col in changed_rows.columns:
                    self.df.at[ci, col] = changed_rows.at[j, col]

        if added_names:
            added_rows = new_df[new_df['name'].isin(set(added_names))].reset_index(drop=True)
            for _, attr, col in _LAYER_SPEC:
                vecs = self._encode_texts(added_rows[col].tolist())
                setattr(self, attr, torch.cat([getattr(self, attr), vecs], dim=0))
            self.df = pd.concat([self.df, added_rows], ignore_index=True)

        self.max_log_count = float(np.log1p(self.df['post_count'].max()))
        self._name_to_idx  = {n: i for i, n in enumerate(self.df['name'])}
        self._tag_names_set = set(self._name_to_idx.keys())
        self._rebuild_arrays_from_df()
        self._normalize_embeddings()
        self._save_cache()
        print(f'[Engine] incremental update done in {time.time() - t0:.2f}s ({len(self.df)} rows)')

    def _save_cache(self) -> None:
        self.paths.ensure_dir()
        st_save(
            {attr: getattr(self, attr).half() for _, attr, _ in _LAYER_SPEC},
            str(self.paths.embeddings),
        )
        save_cols = ['name', 'cn_name', 'cn_core', 'wiki', 'nsfw', 'category', 'post_count']
        self.df[save_cols].to_parquet(str(self.paths.metadata), index=False)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.paths.meta_json, 'w', encoding='utf-8') as f:
            json.dump({
                'schema_version': SCHEMA_VERSION,
                'updated_at':     current_time,
            }, f, ensure_ascii=False, indent=4)
        print(f'[Engine] cache saved ({len(self.df)} rows) at {current_time}')

    def _load_from_cache(self) -> None:
        tensors          = st_load(str(self.paths.embeddings), device=self.device)
        for _, attr, _ in _LAYER_SPEC:
            setattr(self, attr, tensors[attr].float())
        self.df          = pd.read_parquet(str(self.paths.metadata))
        self.max_log_count = float(np.log1p(self.df['post_count'].max()))

    def _cached_schema_version(self) -> int:
        try:
            with open(self.paths.meta_json, 'r', encoding='utf-8') as f:
                return int(json.load(f).get('schema_version', 1))
        except Exception:
            return 0

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode(
            texts, batch_size=64, show_progress_bar=False, convert_to_tensor=True,
        ).float()

    def _load_model(self) -> None:
        if self.model is not None:
            return
        print(f'[Engine] loading model (path={self.model_path}, device={self.device})')
        try:
            self.model = SentenceTransformer(self.model_path, device=self.device)
        except Exception as e:
            print(f'[Engine] model load failed, falling back to HF: {e}')
            self.model = SentenceTransformer(HF_MODEL_ID, device=self.device)

    def _read_csv_robust(self, path: str) -> pd.DataFrame:
        for enc in ['utf-8', 'gbk', 'gb18030']:
            try:
                return pd.read_csv(path, dtype=str, encoding=enc).fillna('')
            except UnicodeDecodeError:
                continue
        raise ValueError(f'failed to read CSV with any known encoding: {path}')

    def _preprocess_raw_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.dropna(subset=['name'], inplace=True)
        df = df[df['name'].str.strip() != '']
        for col in ['cn_name', 'category', 'wiki', 'nsfw']:
            if col not in df.columns:
                df[col] = ''
        df['category'] = df['category'].fillna('0')
        df['nsfw']     = df['nsfw'].fillna('0')
        for char in ['，', '|', '、']:
            df['cn_name'] = df['cn_name'].str.replace(char, ',', regex=False)
        if 'post_count' not in df.columns:
            df['post_count'] = 0
        df['post_count'] = pd.to_numeric(df['post_count'], errors='coerce').fillna(0)
        df['cn_name']    = df['cn_name'].fillna('')
        df['wiki']       = df['wiki'].fillna('')
        df['cn_core']    = df['cn_name'].str.split(',', n=1).str[0].str.strip().fillna('')
        df.drop_duplicates(subset=['name'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _setup_jieba_from_memory(self) -> None:
        if self.df is None:
            return
        unique_words: set[str] = set()
        for text in self.df['cn_name'].dropna().astype(str):
            for part in text.replace(',', ' ').split():
                part = part.strip()
                if len(part) > 1:
                    unique_words.add(part)
        for word in unique_words:
            jieba.add_word(word, 2000)

    def _detect_intent(self, query: str) -> dict[str, float]:
        """根据查询词特征返回各视图的语义分系数。"""
        stripped = query.replace(' ', '')
        cn_chars = sum(1 for c in stripped if '一' <= c <= '鿿')
        en_chars = sum(1 for c in stripped if c.isascii() and c.isalpha())
        total    = max(len(stripped), 1)
        is_long  = len(query) > 8
        is_cn    = cn_chars / total > 0.5
        is_en    = en_chars / total > 0.5

        if is_long and is_cn:
            return {'英文': 0.8, '中文核心词': 0.9, '中文扩展词': 1.1, '释义': 1.4}
        if is_long and is_en:
            return {'英文': 1.0, '中文核心词': 0.8, '中文扩展词': 0.9, '释义': 1.3}
        if is_cn:
            return {'英文': 0.8, '中文核心词': 1.3, '中文扩展词': 1.1, '释义': 0.6}
        if is_en:
            return {'英文': 1.3, '中文核心词': 1.0, '中文扩展词': 0.8, '释义': 0.6}
        return {'英文': 1.0, '中文核心词': 1.0, '中文扩展词': 1.0, '释义': 1.0}

    # ── 查询切分 ──────────────────────────────────────────────────────────

    def _smart_split(self, text: str) -> tuple[list[str], list[str]]:
        """将查询文本拆分为关键词列表，同时返回从句级片段。

        Returns:
            (tokens, segments):
            - tokens: 处理后的关键词列表
            - segments: CN-region 切出的子句片段
        """
        user_pieces = [s.strip() for s in re.split(r'[\s\n\r，、；。]+', text) if s.strip()]
        if not user_pieces:
            return [], []
        has_boundary = len(user_pieces) > 1

        tokens: list[str] = []
        segments: list[str] = []

        cjk_region = r'[一-龥]+(?:[\s\n\r，、；。]+[一-龥]+)*'
        parts = re.split(f'({cjk_region})', text)

        for part in parts:
            if not part.strip():
                continue

            if re.search(r'[一-龥]', part):
                cn_segs = [s.strip() for s in re.split(r'[\s\n\r，、；。]+', part) if s.strip()]
                for seg in cn_segs:
                    segments.append(seg)
                    if has_boundary and re.match(r'^[一-龥]+$', seg) and len(seg) <= _ATOMIC_CJK_MAX_LEN:
                        tokens.append(seg)
                    else:
                        for chunk in re.split(r'([一-龥]+)', seg):
                            if not chunk.strip():
                                continue
                            if re.match(r'[一-龥]+', chunk):
                                tokens.extend(jieba.cut(chunk))
                            else:
                                tokens.extend(self._tokenize_en_chunk(chunk))
            else:
                tokens.extend(self._tokenize_en_chunk(part))

        return tokens, segments

    # ── 英文分词辅助 ──────────────────────────────────────────────────────

    def _tokenize_en_chunk(self, chunk: str) -> list[str]:
        """对一段英文文本做分词：清洗 → 按空格切分 → 过滤停用词/纯数 → 变体规范化 → 合并已知复合标签。"""
        cleaned = re.sub(r'[,()\[\]{}:]', ' ', chunk)
        raw = [p for p in cleaned.split() if p]
        tag_set = getattr(self, '_tag_names_set', None)
        tokens: list[str] = []
        for part in raw:
            low = part.lower()
            if tag_set and low in tag_set:
                tokens.append(low)
                continue
            if low in STOP_WORDS:
                continue
            variant = self._resolve_tag_variant(low)
            if variant:
                tokens.append(variant)
                continue
            if part.isdigit():
                continue
            tokens.append(low)
        if not tokens:
            return []
        return self._merge_compound_english(tokens)

    def _resolve_tag_variant(self, low: str) -> str | None:
        """对未直接命中 tag_set 的英文 token 探测常见变体。"""
        tag_set = getattr(self, '_tag_names_set', None)
        if not tag_set:
            return None

        bases = [low]
        if '-' in low:
            hyphen_normalized = low.replace('-', '_')
            if hyphen_normalized in tag_set:
                return hyphen_normalized
            bases.append(hyphen_normalized)

        for base in bases:
            if base.endswith('s') and len(base) > 1:
                v = base[:-1]
                if v in tag_set:
                    return v
            if base.endswith('es') and len(base) > 2:
                v = base[:-2]
                if v in tag_set:
                    return v
            if base.endswith('ies') and len(base) > 3:
                v = base[:-3] + 'y'
                if v in tag_set:
                    return v
        return None

    def _merge_compound_english(self, tokens: list[str]) -> list[str]:
        """将相邻英文单词合并为已知的 Danbooru 下划线复合标签。"""
        tag_set = getattr(self, '_tag_names_set', None)
        if tag_set is None or len(tokens) < 2:
            return tokens

        result: list[str] = []
        i = 0
        max_w = min(_EN_MAX_COMPOUND, len(tokens))
        while i < len(tokens):
            merged = False
            for w in range(max_w, 1, -1):
                if i + w > len(tokens):
                    continue
                candidate = '_'.join(tokens[i:i + w])
                if candidate in tag_set:
                    result.append(candidate)
                    i += w
                    merged = True
                    break
            if not merged:
                result.append(tokens[i])
                i += 1
        return result

    # ── 关联推荐 ──────────────────────────────────────────────────────────

    def get_related(
        self,
        seed_tags: list[str],
        exclude: set[str] | None = None,
        limit: int = 20,
        show_nsfw: bool = True,
    ) -> list:
        from .models import RelatedTag

        if not self.cooc or not seed_tags:
            return []
        exclude = exclude or set()

        related_key = (tuple(sorted(seed_tags)), tuple(sorted(exclude)), limit, show_nsfw)
        cached = self._related_cache.get(related_key)
        if cached is not None:
            return cached

        total_posts = float(max(self.df['post_count'].max(), 7000000.0))

        npmi_scores: dict[str, float] = {}
        total_cooc:  dict[str, int]   = {}
        tag_sources: dict[str, list[str]] = {}
        name_to_idx = self._name_to_idx
        arr_post_count = self._arr_post_count

        for seed in seed_tags:
            if seed not in name_to_idx:
                continue
            seed_count = float(arr_post_count[name_to_idx[seed]] or 1)

            for neighbor, cnt in self.cooc.get(seed, []):
                if neighbor in exclude or neighbor == seed:
                    continue
                if neighbor not in name_to_idx:
                    continue

                neighbor_count = float(arr_post_count[name_to_idx[neighbor]] or 1)
                cooc = min(float(cnt), seed_count, neighbor_count)
                if cooc <= 0:
                    continue

                numerator = (cooc * total_posts) / (seed_count * neighbor_count)
                if numerator <= 1.0:
                    continue

                pmi   = math.log(numerator)
                p_a_b = cooc / total_posts
                npmi  = 1.0 if p_a_b >= 1.0 else pmi / -math.log(p_a_b)

                npmi_scores[neighbor] = npmi_scores.get(neighbor, 0.0) + npmi
                total_cooc[neighbor]  = total_cooc.get(neighbor, 0) + cnt
                tag_sources.setdefault(neighbor, []).append(seed)

        if not npmi_scores:
            return []

        max_score        = max(npmi_scores.values())
        sorted_candidates = sorted(npmi_scores.items(), key=lambda x: x[1], reverse=True)
        results = []

        for tag_name, raw_score in sorted_candidates:
            if len(results) >= limit:
                break
            idx = name_to_idx[tag_name]
            nsfw = self._arr_nsfw[idx]
            if nsfw == '1' and not show_nsfw:
                continue
            cat = CAT_MAP.get(self._arr_category[idx], 'Other')
            results.append(RelatedTag(
                tag=tag_name,
                cn_name=str(self._arr_cn_name[idx]),
                category=cat,
                nsfw=nsfw,
                cooc_count=total_cooc.get(tag_name, 0),
                cooc_score=round(raw_score / max_score, 4),
                sources=tag_sources.get(tag_name, []),
                post_count=int(self._arr_post_count[idx]),
                wiki=str(self._arr_wiki[idx]) if self._arr_wiki is not None else '',
            ))

        self._related_cache.put(related_key, results)
        return results

    def _load_cooc(self) -> None:
        csv_path     = Path(self.cooc_file)
        parquet_path = csv_path.with_suffix('.parquet')

        if parquet_path.is_file() and (
            not csv_path.is_file()
            or parquet_path.stat().st_mtime >= csv_path.stat().st_mtime
        ):
            read_path  = parquet_path
            is_parquet = True
        elif csv_path.is_file():
            read_path  = csv_path
            is_parquet = False
        else:
            print(f'[Engine] co-occurrence table not found ({self.cooc_file}), related tags disabled')
            return

        print(f'[Engine] loading co-occurrence table ({read_path.name})...')
        t0 = time.time()
        try:
            if is_parquet:
                df = pd.read_parquet(str(read_path))
            else:
                df = self._read_csv_robust(str(read_path))
                df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
                df.to_parquet(str(parquet_path), index=False)
                print(f'[Engine] co-occurrence table cached as {parquet_path.name}')

            tag_a  = df['tag_a'].astype(str).to_numpy()
            tag_b  = df['tag_b'].astype(str).to_numpy()
            counts = df['count'].astype(int).to_numpy()

            src = np.concatenate([tag_a, tag_b])
            dst = np.concatenate([tag_b, tag_a])
            cnt = np.concatenate([counts, counts])

            sort_idx = np.lexsort((-cnt, src))
            src = src[sort_idx]
            dst = dst[sort_idx]
            cnt = cnt[sort_idx]

            unique_srcs, first_pos = np.unique(src, return_index=True)
            end_pos = np.append(first_pos[1:], len(src))

            cooc: dict[str, list[tuple[str, int]]] = {}
            for s, start, end in zip(unique_srcs, first_pos, end_pos):
                cooc[s] = list(zip(dst[start:end].tolist(), cnt[start:end].tolist()))

            self.cooc = cooc
            print(
                f'[Engine] co-occurrence table loaded: {len(cooc):,} tags, '
                f'{time.time() - t0:.2f}s'
            )
        except Exception as e:
            print(f'[Engine] failed to load co-occurrence table: {e}')
