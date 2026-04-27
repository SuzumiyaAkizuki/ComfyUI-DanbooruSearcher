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
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import jieba
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file as st_load, save_file as st_save
from sentence_transformers import SentenceTransformer, util

from .models import SearchRequest, SearchResponse, TagResult


STOP_WORDS: frozenset[str] = frozenset({
    ',', '.', ':', ';', '?', '!', '"', "'", '`',
    '(', ')', '[', ']', '{', '}', '<', '>',
    '-', '_', '=', '+', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '~',
    '，', '。', '：', '；', '？', '！', '\u201c', '\u201d', '\u2018', '\u2019',
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
})

CAT_MAP: dict[str, str] = {
    '0': 'General', '1': 'Artist', '3': 'Copyright', '4': 'Character', '5': 'Meta',
}

LOCAL_MODEL_PATH = 'my_model_bge_m3'
HF_MODEL_ID      = 'BAAI/bge-m3'
SCHEMA_VERSION   = 2


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
        self.is_loaded = True
        print(f'[Engine] ready in {time.time() - t0:.2f}s')

    def search(self, request: SearchRequest) -> SearchResponse:
        if not self.is_loaded:
            self.load()

        if request.use_segmentation:
            raw_kw   = self._smart_split(request.query)
            keywords = [w.strip() for w in raw_kw if w.strip() and w.strip() not in STOP_WORDS]
            queries  = [request.query] + keywords
        else:
            keywords = []
            queries  = [request.query]

        q_emb = self.model.encode(queries, convert_to_tensor=True, show_progress_bar=False).float()
        empty = [[] for _ in queries]
        tl    = request.target_layers
        k     = request.top_k

        layer_weights = self._detect_intent(request.query)
        active_layers = [l for l in ['英文', '中文扩展词', '释义', '中文核心词'] if l in tl]
        if active_layers:
            aw       = {l: layer_weights.get(l, 1.0) for l in active_layers}
            total_aw = sum(aw.values())
            pvk      = {l: max(1, round(k * aw[l] / total_aw)) for l in active_layers}
        else:
            pvk = {}

        hits_en   = util.semantic_search(q_emb, self.emb_en,      top_k=pvk.get('英文',      1)) if '英文'      in tl else empty
        hits_cn   = util.semantic_search(q_emb, self.emb_cn,      top_k=pvk.get('中文扩展词', 1)) if '中文扩展词' in tl else empty
        hits_wiki = util.semantic_search(q_emb, self.emb_wiki,    top_k=pvk.get('释义',       1)) if '释义'      in tl else empty
        hits_core = util.semantic_search(q_emb, self.emb_cn_core, top_k=pvk.get('中文核心词', 1)) if '中文核心词' in tl else empty

        final: dict[str, TagResult] = {}

        for i, source_word in enumerate(queries):
            combined = (
                [(h, '英文')        for h in hits_en[i]]
                + [(h, '中文扩展词') for h in hits_cn[i]]
                + [(h, '释义')       for h in hits_wiki[i]]
                + [(h, '中文核心词') for h in hits_core[i]]
            )
            for hit, layer in combined:
                score = hit['score']
                if score < 0.35:
                    continue
                idx      = hit['corpus_id']
                row      = self.df.iloc[idx]
                cat_text = CAT_MAP.get(str(row.get('category', '0')), 'Other')
                if cat_text not in request.target_categories:
                    continue
                tag_name    = row['name']
                count       = row['post_count']
                pop_score   = np.log1p(count) / self.max_log_count
                w           = request.popularity_weight
                final_score = score * layer_weights.get(layer, 1.0) * (1 - w) + pop_score * w
                if tag_name not in final or final_score > final[tag_name].final_score:
                    final[tag_name] = TagResult(
                        tag=tag_name, cn_name=row['cn_name'], category=cat_text,
                        nsfw=str(row.get('nsfw', '0')),
                        final_score=round(float(final_score), 4),
                        semantic_score=round(float(score), 4),
                        count=int(count), source=source_word, layer=layer,
                        wiki=str(row.get('wiki', '')),
                    )

        # Guarantee the top-1 result per query source (above threshold).
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
        return SearchResponse(
            tags_all=tags_all, tags_sfw=tags_sfw,
            results=valid, keywords=keywords,
        )

    def _build_full(self) -> None:
        print(f'[Engine] reading {self.csv_path}')
        raw_df             = self._read_csv_robust(self.csv_path)
        self.df            = self._preprocess_raw_df(raw_df)
        self.max_log_count = float(np.log1p(self.df['post_count'].max()))
        self._encode_all_and_save()

    def _encode_all_and_save(self) -> None:
        print('[Engine] encoding all rows...')
        self.emb_en      = self._encode_texts(self.df['name'].tolist())
        self.emb_cn      = self._encode_texts(self.df['cn_name'].tolist())
        self.emb_wiki    = self._encode_texts(self.df['wiki'].tolist())
        self.emb_cn_core = self._encode_texts(self.df['cn_core'].tolist())
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
            self.df          = self.df[keep_mask].reset_index(drop=True)
            self.emb_en      = self.emb_en[keep_pos]
            self.emb_cn      = self.emb_cn[keep_pos]
            self.emb_wiki    = self.emb_wiki[keep_pos]
            self.emb_cn_core = self.emb_cn_core[keep_pos]
            cached_idx = {n: i for i, n in enumerate(self.df['name'])}

        if changed_names:
            changed_rows = new_df[new_df['name'].isin(set(changed_names))].reset_index(drop=True)
            vecs_en   = self._encode_texts(changed_rows['name'].tolist())
            vecs_cn   = self._encode_texts(changed_rows['cn_name'].tolist())
            vecs_wiki = self._encode_texts(changed_rows['wiki'].tolist())
            vecs_core = self._encode_texts(changed_rows['cn_core'].tolist())
            for j, name in enumerate(changed_rows['name']):
                ci = cached_idx[name]
                self.emb_en[ci]      = vecs_en[j]
                self.emb_cn[ci]      = vecs_cn[j]
                self.emb_wiki[ci]    = vecs_wiki[j]
                self.emb_cn_core[ci] = vecs_core[j]
                for col in changed_rows.columns:
                    self.df.at[ci, col] = changed_rows.at[j, col]

        if added_names:
            added_rows = new_df[new_df['name'].isin(set(added_names))].reset_index(drop=True)
            vecs_en   = self._encode_texts(added_rows['name'].tolist())
            vecs_cn   = self._encode_texts(added_rows['cn_name'].tolist())
            vecs_wiki = self._encode_texts(added_rows['wiki'].tolist())
            vecs_core = self._encode_texts(added_rows['cn_core'].tolist())
            self.emb_en      = torch.cat([self.emb_en,      vecs_en],   dim=0)
            self.emb_cn      = torch.cat([self.emb_cn,      vecs_cn],   dim=0)
            self.emb_wiki    = torch.cat([self.emb_wiki,    vecs_wiki], dim=0)
            self.emb_cn_core = torch.cat([self.emb_cn_core, vecs_core], dim=0)
            self.df = pd.concat([self.df, added_rows], ignore_index=True)

        self.max_log_count = float(np.log1p(self.df['post_count'].max()))
        self._name_to_idx  = {n: i for i, n in enumerate(self.df['name'])}
        self._save_cache()
        print(f'[Engine] incremental update done in {time.time() - t0:.2f}s ({len(self.df)} rows)')

    def _save_cache(self) -> None:
        self.paths.ensure_dir()
        st_save(
            {
                'emb_en':      self.emb_en.half(),
                'emb_cn':      self.emb_cn.half(),
                'emb_wiki':    self.emb_wiki.half(),
                'emb_cn_core': self.emb_cn_core.half(),
            },
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
        self.emb_en      = tensors['emb_en'].float()
        self.emb_cn      = tensors['emb_cn'].float()
        self.emb_wiki    = tensors['emb_wiki'].float()
        self.emb_cn_core = tensors['emb_cn_core'].float()
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
        """
        Infer per-layer weight coefficients from the query's language characteristics.
        Values > 1.0 boost a layer; < 1.0 down-weight it.
        """
        stripped = query.replace(' ', '')
        cn_chars = sum(1 for c in stripped if '\u4e00' <= c <= '\u9fff')
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

    def _smart_split(self, text: str) -> list[str]:
        tokens: list[str] = []
        for chunk in re.split(r'([一-龥]+)', text):
            if not chunk.strip():
                continue
            if re.match(r'[一-龥]+', chunk):
                tokens.extend(jieba.cut(chunk))
            else:
                cleaned = re.sub(r'[,()\[\]{}:]', ' ', chunk)
                for part in cleaned.split():
                    try:
                        float(part)
                    except ValueError:
                        tokens.append(part)
        return tokens

    def get_related(
        self,
        seed_tags: list[str],
        exclude: set[str] | None = None,
        limit: int = 20,
        show_nsfw: bool = True,
    ) -> list:
        from .models import RelatedTag
        import math

        if not self.cooc or not seed_tags:
            return []
        exclude = exclude or set()

        total_posts = float(max(self.df['post_count'].max(), 7000000.0))

        npmi_scores: dict[str, float] = {}
        total_cooc:  dict[str, int]   = {}
        tag_sources: dict[str, list[str]] = {}
        name_to_idx = self._name_to_idx

        for seed in seed_tags:
            if seed not in name_to_idx:
                continue
            seed_count = float(self.df.iloc[name_to_idx[seed]].get('post_count', 1) or 1)

            for neighbor, cnt in self.cooc.get(seed, []):
                if neighbor in exclude or neighbor == seed:
                    continue
                if neighbor not in name_to_idx:
                    continue

                neighbor_count = float(
                    self.df.iloc[name_to_idx[neighbor]].get('post_count', 1) or 1
                )
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
            row  = self.df.iloc[name_to_idx[tag_name]]
            nsfw = str(row.get('nsfw', '0'))
            if nsfw == '1' and not show_nsfw:
                continue
            cat = CAT_MAP.get(str(row.get('category', '0')), 'Other')
            results.append(RelatedTag(
                tag=tag_name,
                cn_name=str(row.get('cn_name', '')),
                category=cat,
                nsfw=nsfw,
                cooc_count=total_cooc[tag_name],
                cooc_score=round(raw_score / max_score, 4),
                sources=tag_sources.get(tag_name, []),
            ))

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