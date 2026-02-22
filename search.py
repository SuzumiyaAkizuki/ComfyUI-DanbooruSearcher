import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import pickle
import jieba
import numpy as np
import re
import time


# ================= 核心算法类 (DanbooruTagger) =================
class DanbooruTagger:
    _instance = None  # 单例模式

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DanbooruTagger, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def init_engine(self, model_path, cache_path, csv_path):
        """
        初始化引擎，包含首次自动构建逻辑
        """
        if self.initialized:
            return

        self.cache_path = cache_path
        self.csv_path = csv_path
        self.model_path = model_path if model_path else 'BAAI/bge-m3'

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.df = None

        # 四层索引
        self.emb_en = None
        self.emb_cn = None
        self.emb_wiki = None
        self.emb_cn_core = None

        self.max_log_count = 15.0

        # 停用词表
        self.stop_words = {
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
            '有', '无', '非', '没', '不'
        }

        self._load()
        self.initialized = True

    def _load(self):
        # 自动调整缓存路径以匹配设备
        base, ext = os.path.splitext(self.cache_path)
        real_cache = f"{base}_{self.device}_fp16{ext}"

        if not os.path.exists(real_cache):
            print(f"[DanbooruSearch] Cache not found. Building from CSV: {self.csv_path}")
            self._build_from_csv(real_cache)
        else:
            print(f"[DanbooruSearch] Loading cache: {real_cache}")
            try:
                with open(real_cache, 'rb') as f:
                    data = pickle.load(f)
                    self.df = data['df']
                    self.emb_en = data['embeddings_en'].float()
                    self.emb_cn = data['embeddings_cn'].float()
                    self.emb_wiki = data.get('embeddings_wiki', torch.zeros_like(self.emb_en)).float()
                    self.emb_cn_core = data.get('embeddings_cn_core', torch.zeros_like(self.emb_en)).float()
                    self.max_log_count = data.get('max_log_count', 15.0)
            except Exception as e:
                print(f"[DanbooruSearch] Cache load failed ({e}), rebuilding...")
                self._build_from_csv(real_cache)

        if self.model is None:
            print(f"[DanbooruSearch] Loading Model: {self.model_path}")
            self.model = SentenceTransformer(self.model_path, device=self.device)

        # 内存构建 Jieba 字典
        print("[DanbooruSearch] Building Jieba dict...")
        if self.df is not None and 'cn_name' in self.df.columns:
            all_text = self.df['cn_name'].dropna().astype(str)
            for text in all_text:
                parts = text.replace(',', ' ').split()
                for part in parts:
                    if len(part) > 1: jieba.add_word(part.strip(), 2000)

    def _read_csv_robust(self, path):
        encodings = ['utf-8', 'gbk', 'gb18030']
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, encoding=enc).fillna("")
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"[DanbooruSearch] CSV Error ({enc}): {e}")
                break
        raise ValueError(f"无法读取 CSV 文件: {path}，请检查文件是否存在或编码格式。")

    def _build_from_csv(self, save_path):
        df = self._read_csv_robust(self.csv_path)

        if 'post_count' not in df.columns: df['post_count'] = 0
        df['post_count'] = pd.to_numeric(df['post_count'], errors='coerce').fillna(0)

        if 'cn_name' not in df.columns: df['cn_name'] = ""
        if 'wiki' not in df.columns: df['wiki'] = ""
        if 'name' not in df.columns:
            raise ValueError("CSV missing 'name' column")

        df['cn_core'] = df['cn_name'].str.split(',', n=1).str[0].str.strip()

        print("[DanbooruSearch] Encoding embeddings (this may take a while)...")
        if self.model is None:
            self.model = SentenceTransformer(self.model_path, device=self.device)

        self.emb_en = self.model.encode(df['name'].tolist(), batch_size=64, show_progress_bar=True,
                                        convert_to_tensor=True).float()
        self.emb_cn = self.model.encode(df['cn_name'].tolist(), batch_size=64, show_progress_bar=True,
                                        convert_to_tensor=True).float()
        self.emb_wiki = self.model.encode(df['wiki'].tolist(), batch_size=64, show_progress_bar=True,
                                          convert_to_tensor=True).float()
        self.emb_cn_core = self.model.encode(df['cn_core'].tolist(), batch_size=64, show_progress_bar=True,
                                             convert_to_tensor=True).float()

        cache_data = {
            'df': df,
            'embeddings_en': self.emb_en.half(),
            'embeddings_cn': self.emb_cn.half(),
            'embeddings_wiki': self.emb_wiki.half(),
            'embeddings_cn_core': self.emb_cn_core.half(),
            'max_log_count': np.log1p(df['post_count'].max())
        }

        with open(save_path, 'wb') as f:
            pickle.dump(cache_data, f)

        self.df = df
        print("[DanbooruSearch] Cache built successfully.")

    def _smart_split(self, text):
        tokens = []
        chunks = re.split(r'([\u4e00-\u9fa5]+)', text)
        for chunk in chunks:
            if not chunk.strip(): continue
            if re.match(r'[\u4e00-\u9fa5]+', chunk):
                tokens.extend(jieba.cut(chunk))
            else:
                parts = re.sub(r'[,()\[\]{}:]', ' ', chunk).split()
                tokens.extend([p for p in parts if not p.isdigit()])
        return tokens

    def search(self, user_query, top_k=5, limit=80, popularity_weight=0.15):
        if self.df is None:
            return "", "Error: Database not loaded."

        # 构建查询序列
        keywords = self._smart_split(user_query)
        search_queries = [user_query] + [w for w in keywords if w not in self.stop_words]

        # 编码查询
        query_embeddings = self.model.encode(search_queries, convert_to_tensor=True).float()

        # 多层检索
        hits_en = util.semantic_search(query_embeddings, self.emb_en, top_k=top_k)
        hits_cn = util.semantic_search(query_embeddings, self.emb_cn, top_k=top_k)
        hits_wiki = util.semantic_search(query_embeddings, self.emb_wiki, top_k=top_k)
        hits_cn_core = util.semantic_search(query_embeddings, self.emb_cn_core, top_k=top_k)

        final_results = {}
        for i, _ in enumerate(search_queries):
            source_word = search_queries[i]
            # 标记来源层
            combined = [(h, 'EN') for h in hits_en[i]] + [(h, 'CN') for h in hits_cn[i]] + \
                       [(h, 'Wiki') for h in hits_wiki[i]] + [(h, 'Core') for h in hits_cn_core[i]]

            for hit, layer in combined:
                idx = hit['corpus_id']
                score = hit['score']
                row = self.df.iloc[idx]
                tag_name = row['name']

                pop_score = np.log1p(float(row['post_count'])) / self.max_log_count
                final_score = (score * (1 - popularity_weight)) + (pop_score * popularity_weight)

                if tag_name not in final_results or final_score > final_results[tag_name]['final_score']:
                    final_results[tag_name] = {
                        'tag': tag_name,
                        'final_score': final_score,
                        'semantic_score': score,  # 保留语义分
                        'source': source_word,  # 保留匹配来源词
                        'cn_name': row['cn_name'],
                        'layer': layer
                    }

        sorted_tags = sorted(final_results.values(), key=lambda x: x['final_score'], reverse=True)[:limit]
        tags_string = ", ".join([item['tag'] for item in sorted_tags])

        # 生成原始格式的 Debug 表格
        debug_lines = []
        debug_lines.append(f"{'匹配标签':<25} | {'综合分':<6} | {'语义分':<6} | {'来源':<8} | {'中文含义'}")
        debug_lines.append("-" * 100)

        for item in sorted_tags:
            score_str = f"{item['final_score']:.3f}"
            sem_str = f"{item['semantic_score']:.3f}"
            source_str = str(item['source'])[:8]  # 截断一下避免太长
            debug_lines.append(
                f"{item['tag']:<28} | {score_str:<6} | {sem_str:<6}  | {source_str:<10} | {item['cn_name']}"
            )

        return tags_string, "\n".join(debug_lines)



class DanbooruTagSearch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"multiline": False, "default": ""}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "limit": ("INT", {"default": 80, "min": 10, "max": 300, "step": 10}),
                "popularity_weight": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05}),
                "text": ("STRING", {"multiline": True, "default": "一个在雨中奔跑的女孩"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tags_string", "debug_info")
    FUNCTION = "search"
    CATEGORY = "utils/prompt"

    def search(self, text, model_path, top_k, limit, popularity_weight):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = os.path.join(current_dir, "tags_enhanced.csv")
        if not os.path.exists(csv_file): csv_file = os.path.join(current_dir, "tags.csv")

        cache_file = os.path.join(current_dir, "danbooru_vectors_multiview.pkl")

        tagger = DanbooruTagger()
        tagger.init_engine(model_path, cache_file, csv_file)

        tags, info = tagger.search(
            user_query=text, top_k=top_k, limit=limit, popularity_weight=popularity_weight
        )
        return (tags, info)