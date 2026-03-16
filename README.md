# ComfyUI-DanbooruSearcher

![](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202603161624413.png)

本插件提供一种在 ComfyUI 中模糊查找 Danbooru 标签的方法。用户可以凭借模糊的自然语言描述，查找 Danbooru 数据集中对应的标签。

在线试用：[DanbooruSearch on HuggingFace](https://huggingface.co/spaces/SAkizuki/DanbooruSearch)

---

## 安装

1. 将本仓库克隆到 `ComfyUI/custom_nodes/` 目录下：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/SuzumiyaAkizuki/ComfyUI-DanbooruSearcher
   ```

2. 安装依赖：

   ```bash
   pip install -r ComfyUI-DanbooruSearcher/requirements.txt
   ```

3. 重启 ComfyUI。

---

## 准备模型

本插件需要 [BGE-M3](https://huggingface.co/BAAI/bge-m3) 向量模型。

**推荐**：提前下载到本地，在节点的 `model_path` 参数中填写本地路径。若留空，插件会在首次运行时自动从 HuggingFace 下载（需要网络连接）。

本地模型的目录结构示例：

```
D:\Models\bge-m3\
│  config.json
│  pytorch_model.bin
│  tokenizer.json
│  ...
└─1_Pooling\
       config.json
```

在 `model_path` 中填写该目录路径即可：

```
D:\Models\bge-m3
```

---

## 节点说明

### Danbooru Smart Search

语义搜索节点。输入自然语言描述，返回匹配的 Danbooru 标签候选池。

**输入参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text` | STRING | — | 用户输入的自然语言描述，支持中文和英文 |
| `model_path` | STRING | 空 | BGE-M3 模型本地路径，留空则自动下载 |
| `top_k` | INT | 5 | 每个分词各取前 top_k 个候选标签 |
| `limit` | INT | 80 | 最终输出的标签数量上限 |
| `popularity_weight` | FLOAT | 0.15 | 标签热度在综合得分中的权重（0~1），推荐 0.15 |
| `use_segmentation` | BOOLEAN | True | 开启后对输入分词后分别检索，适合完整句子描述；关闭后整句检索，适合精准查找单个词 |
| `show_nsfw` | BOOLEAN | True | 是否在结果中包含 NSFW 标签 |

**输出：**

| 输出 | 类型 | 说明 |
|------|------|------|
| `search_result` | Python对象：DANBOORU_RESULT | 搜索结果对象，可连接到 Danbooru Related Tags 节点 |
| `tags_string` | STRING | 逗号分隔的标签字符串，可直接连接到下游节点 |
| `debug_info` | STRING | 搜索详细信息，包含综合分、语义分、来源词、中文含义 |

---

### Danbooru Related Tags

共现推荐节点。基于搜索结果，从共现数据库中查找关联标签，扩充候选标签池。

本节点的设计目标是作为 **RAG（检索增强生成）** 的检索层使用——将扩充后的候选池传给 LLM，让 LLM 在给定的标签集合中选择，而非自由生成，从而避免 LLM 编造 Danbooru 上不存在的标签。

**输入参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `search_result` | Python对象：DANBOORU_RESULT | — | 接收 Danbooru Smart Search 的 `search_result` 输出 |
| `limit` | INT | 50 | 共现推荐的标签数量上限 |
| `show_nsfw` | BOOLEAN | True | 是否在推荐结果中包含 NSFW 标签 |

**输出：**

| 输出 | 类型 | 说明 |
|------|------|------|
| `related_tags` | STRING | 仅共现推荐的标签（不含搜索结果） |
| `combined_tags` | STRING | 搜索结果 + 共现推荐合并去重，适合直接传给 LLM |

**推荐种子选取策略：** 节点从搜索结果中，按分词自动取每个词语义得分最高的一条作为种子，而非使用全部搜索结果，以保证推荐的精准度。

---

## 使用场景

### 场景一：完整画面描述 → 标签集

输入对画面的完整描述，节点将生成匹配的 Danbooru 标签集，推荐参数：`top_k=5`，`limit=80`，`use_segmentation=True`。

**示例输入：**

```
一个穿着白色水手服、蓝色短裙的少女，在下着大雨的城市街道奔跑，她的表情是不甘、愤怒、流泪，她的衣服湿透。
```

**示例输出（tags_string）：**

```
short_dress, city, streaming_tears, street, white_serafuku, rain, tears, miniskirt, white_sailor_collar, serafuku, running, wet_clothes, wet_face, angry, crying, urban, wet_hair, cityscape, outdoors, after_rain ...
```

---

### 场景二：关键词精准查找

对某个概念有模糊印象但不知道对应的 Danbooru 标签，推荐参数：`top_k=5`，`limit=10`，`use_segmentation=False`。

**示例输入：**

```
假肢
```

**示例输出（tags_string）：**

```
running_blades, prosthetic_leg, peg_leg, severed_limb, fake_claws, fake_nails, detached_legs, multiple_legs, separated_legs, alternate_footwear
```

---

### 场景三：概念扩展查找

查找某个概念下的全部相关标签，推荐参数：`top_k=40`，`limit=80`，`use_segmentation=False`。

**示例输入：**

```
裙子
```

**示例输出（tags_string）：**

```
skirt, dress, one_piece, white_skirt, black_skirt, upskirt, long_skirt, long_dress, skirt_lift, sweater_dress, gown, wet_dress, armored_skirt, culottes ...
```

---

### 场景四：配合 LLM 使用（推荐工作流）

本插件的核心设计场景是与 LLM 节点联动，解决 LLM 编造不存在标签的问题。例如，用户输入「水手服」，LLM 可能会将其翻译为 `sailor suit`，而「水手服」在 Danbooru 上真正的标签是 `serafuku`。

通过将本插件的输出作为候选标签池传给 LLM，让 LLM 在给定集合中选择，可以显著减少标签幻觉。

推荐与 [ComfyUI-NewBie-LLM-Formatter](https://github.com/SuzumiyaAkizuki/ComfyUI-NewBie-LLM-Formatter) 联动使用：

```
用户描述
  └→ Danbooru Smart Search
       ├→ tags_string ──────────────────────────────→ LLM Formatter (参考候选池)
       └→ search_result → Danbooru Related Tags
                              └→ combined_tags ────→ LLM Formatter (扩充候选池)
```

> **注意**：如果直接将 `tags_string` 连接到 CLIP 编码层，效果可能并不理想，因为搜索结果未经 LLM 筛选，噪声较多。

---

## 注意事项

- 仅支持中文和英文输入
- 仅收录 Danbooru 频数 ≥ 100 的标签
- 仅包含 General、Character、Copyright 三类标签
- 首次运行时需要构建向量缓存（约 1~3 分钟），缓存完成后后续启动速度正常
- 查找结果可能包含 NSFW 内容，可通过 `show_nsfw` 参数关闭