import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import pickle
import jieba
import numpy as np
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
        self.model_path = model_path

        self.model = None
        self.df = None
        self.emb_cn = None
        self.emb_en = None
        self.max_log_count = 15.0

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
        t0 = time.time()

        # 1. 如果没有缓存，则触发自动构建
        if not os.path.exists(self.cache_path):
            print("\n" + "=" * 50)
            print("[DTOOL]: 正在首次构建向量索引库，这可能需要 1~3 分钟...")
            print("[DTOOL]: (此过程只会在第一次运行时执行一次)")
            self._build_from_csv()
        else:
            print(f"[DTOOL]: 正在从 {self.cache_path} 读取索引...")
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                self.df = data['df']
                self.emb_cn = data['embeddings_cn']
                self.emb_en = data['embeddings_en']
                self.max_log_count = data.get('max_log_count', 15.0)

        # 2. 确保模型已被加载 (如果从缓存读取，模型还是None，需要在这里加载)
        if self.model is None:
            print(f"DTOOL: Loading Model from {self.model_path}...")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Missing Model Folder: {self.model_path}")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer(self.model_path, device=device)

        # 3. 初始化 Jieba
        print("DTOOL: Building Jieba Dict from memory...")
        self._setup_jieba_from_memory()

        print(f"DTOOL: Initialization finished in {time.time() - t0:.2f}s")

    def _build_from_csv(self):
        """
        从 CSV 读取数据并使用 GPU/CPU 生成向量库
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Missing Source File: {self.csv_path}\nPlease provide tags.csv!")

        print(f"DTOOL: Reading {self.csv_path} ...")
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8', dtype=str)
        except:
            df = pd.read_csv(self.csv_path, encoding='gbk', dtype=str)

        # 数据清洗
        df.dropna(subset=['name'], inplace=True)
        df = df[df['name'].str.strip() != '']
        if 'cn_name' not in df.columns: df['cn_name'] = ''
        df['cn_name'] = df['cn_name'].fillna('')
        if 'post_count' not in df.columns: df['post_count'] = 0
        df['post_count'] = pd.to_numeric(df['post_count'], errors='coerce').fillna(0)

        self.df = df
        self.max_log_count = np.log1p(df['post_count'].max())

        # 加载模型用于编码
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"DTOOL: Loading Model for encoding (Device: {device})...")
        self.model = SentenceTransformer(self.model_path, device=device)

        # 编码中文层
        print("DTOOL: Encoding Chinese tags...")
        cn_sentences = df['cn_name'].apply(lambda x: x if len(str(x).strip()) > 0 else "UNK").tolist()
        self.emb_cn = self.model.encode(cn_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

        # 编码英文层
        print("DTOOL: Encoding English tags...")
        en_sentences = df['name'].tolist()
        self.emb_en = self.model.encode(en_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

        # 保存到 pkl
        print("DTOOL: Saving cache to disk...")
        cache_data = {
            'df': df,
            'embeddings_cn': self.emb_cn,
            'embeddings_en': self.emb_en,
            'max_log_count': self.max_log_count
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print("DTOOL: Cache built successfully!")

    def _setup_jieba_from_memory(self):
        if self.df is not None:
            cn_tags = self.df['cn_name'].dropna().astype(str).tolist()
            count = 0
            for tag in cn_tags:
                tag = tag.strip()
                if len(tag) > 1:
                    jieba.add_word(tag, 2000)
                    count += 1
            print(f"DTOOL: Added {count} words to Jieba from data.")

    def search(self, user_query, top_k=5, limit=80, popularity_weight=0.15):
        seg_list = list(jieba.cut(user_query))
        keywords = [w.strip() for w in seg_list if w.strip() and w.strip() not in self.stop_words]
        search_queries = [user_query] + keywords

        query_embeddings = self.model.encode(search_queries, convert_to_tensor=True)

        hits_cn = util.semantic_search(query_embeddings, self.emb_cn, top_k=top_k)
        hits_en = util.semantic_search(query_embeddings, self.emb_en, top_k=top_k)

        final_results = {}

        for i, _ in enumerate(search_queries):
            source_word = search_queries[i]
            combined = []
            for h in hits_cn[i]: combined.append((h, 'CN'))
            for h in hits_en[i]: combined.append((h, 'EN'))

            for hit, layer in combined:
                score = hit['score']
                if score < 0.35: continue

                idx = hit['corpus_id']
                row = self.df.iloc[idx]
                tag_name = row['name']
                count = row['post_count']

                log_count = np.log1p(count)
                pop_score = log_count / self.max_log_count
                final_score = (score * (1 - popularity_weight)) + (pop_score * popularity_weight)

                if tag_name not in final_results or final_score > final_results[tag_name]['final_score']:
                    final_results[tag_name] = {
                        'final_score': final_score,
                        'semantic_score': score,
                        'cn_name': row['cn_name'],
                        'count': int(count),
                        'source': source_word,
                        'layer': layer
                    }

        sorted_tags = sorted(final_results.items(), key=lambda x: x[1]['final_score'], reverse=True)
        valid_tags = [item for item in sorted_tags if item[1]['final_score'] > 0.45]

        debug_lines = []
        debug_lines.append(f"{'匹配标签':<25} | {'综合分':<6} | {'语义分':<6} | {'来源':<8} | {'中文含义'}")
        debug_lines.append("-" * 100)

        if len(valid_tags) > limit:
            debug_lines.append(f"[提示] 结果过多 ({len(valid_tags)}个)，已截取前 {limit} 个。")
            valid_tags = valid_tags[:limit]

        output_tags = []
        for tag, info in valid_tags:
            score_str = f"{info['final_score']:.3f}"
            debug_lines.append(
                f"{tag:<28} | {score_str:<6} | {info['semantic_score']:.3f}  | {info['source']:<10} | {info['cn_name']}")
            output_tags.append(tag)

        return ", ".join(output_tags), "\n".join(debug_lines)


# ================= ComfyUI 节点类 =================

class DanbooruTagSearch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path":("STRING",{"multiline":False}),
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

    def search(self, text,model_path, top_k, limit, popularity_weight):
        # 1. 确定资源路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = os.path.join(current_dir, "tags.csv")  # [新增] 传入 CSV 路径
        cache_file = os.path.join(current_dir, "danbooru_vectors_dual.pkl")
        model_dir = model_path

        # 2. 初始化单例
        tagger = DanbooruTagger()

        if not tagger.initialized:
            print(f"DTOOL: Initializing Danbooru Tagger...")
            tagger.init_engine(model_dir, cache_file, csv_file)  # [修改] 传递 3 个参数

        # 3. 执行搜索
        tags, info = tagger.search(text, top_k, limit, popularity_weight)

        print(f"\n[Danbooru Search] Input: {text}")
        print(info)

        return (tags, info)