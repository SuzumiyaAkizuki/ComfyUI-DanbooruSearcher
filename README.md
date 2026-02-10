# ComfyUI-DanbooruSearcher

![](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202602100945958.png)

本项目提供一种模糊查找 Danbooru 标签的方法。用户可以凭借模糊的语言描述，查找 Danbooru 数据集中对应的标签。

在线试用链接：[在线试用](https://huggingface.co/spaces/SAkizuki/DanbooruSearch)


## 节点说明

本节点利用预训练的向量模型将用户输入的自然语言描述映射到高维向量空间，并与标签向量库进行相似度匹配，最终结合标签热度进行混合排序推荐。

![image-20260209214519225](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202602092145168.png)

本节点需要5个输入参数：

- `model_path`：保存向量模型的路径。推荐使用BGE-M3模型。其在HuggingFace上的标识为`BAAI/bge-m3`，下载地址为[BGE-M3](https://huggingface.co/BAAI/bge-m3/tree/main)。

  例如，如果文件夹结构为

  ```powershell
  PS D:\Produce\StableDiffusion\test-coding\my_model_bge_m3> tree /F
  卷 Software 的文件夹 PATH 列表
  卷序列号为 3630-50D1
  D:.
  │  colbert_linear.pt
  │  config.json
  │  config_sentence_transformers.json
  │  long.jpg
  │  modules.json
  │  pytorch_model.bin
  │  README.md
  │  sentencepiece.bpe.model
  │  sentence_bert_config.json
  │  sparse_linear.pt
  │  special_tokens_map.json
  │  tokenizer.json
  │  tokenizer_config.json
  │
  ├─1_Pooling
  │      config.json
  │
  ├─imgs
  │      .DS_Store
  │      bm25.jpg
  │      long.jpg
  │      miracl.jpg
  │      mkqa.jpg
  │      nqa.jpg
  │      others.webp
  │
  └─onnx
          config.json
          Constant_7_attr__value
          model.onnx
          model.onnx_data
          sentencepiece.bpe.model
          special_tokens_map.json
          tokenizer.json
          tokenizer_config.json
  ```

  则在`model_path`中填写：

  ```
  D:\Produce\StableDiffusion\test-coding\my_model_bge_m3
  ```

- `top_k`：每个词取前`top_k`个相关的标签。

- `limit`：对于整个查找，展示综合得分的前`limit`名。

- `popularity_weight`：0~1，这个数值越高，标签频数在综合得分中的权重越大。推荐设为0.15

- `text`：用户输入文本

本节点有2个输出流：

- `tags_string`：字符串格式，输出的标签列表
- `debug_info`：字符串格式，搜索的详细结果

## 主要功能

### 完整画面查找

你可以输入对画面的完整描述，节点将生成匹配的danbooru标签集。

**示例输入**

```
一个穿着白色水手服、蓝色短裙的少女，在下着大雨的城市街道奔跑，她的表情是不甘、愤怒、流泪，她的衣服湿透。
```

**建议参数**

- `top_k`：较小，例如5

- `结果上限`：一般的SDXL系生图模型支持70-80个标签，设为80

- `热度权重`：0.15

**示例输出**

tags_string：

```
short_dress, city, streaming_tears, street, white_serafuku, tag, hashtag, clothes, furious, fume, label, rain, blue_background, tears, miniskirt, white_sailor_collar, shorts_under_skirt, dressing, serafuku, costume, shorts_under_dress, dress, crying, blue_skin, undressing, microskirt, object_in_clothes, tearing_up, blue_theme, black_serafuku, running, blue_shoes, >:(, shared_clothes, clothes_on_and_off, blue_liquid, jogging, clothes_in_front, wiping_tears, grey_serafuku, urban, angry, shorts, fleeing, town, wet_clothes, la_grondement_du_haine, wet_face, slug_girl, price_tag, changing_clothes, clothes_on_floor, angry_sex, wet_shirt, training_bra, wet_hair, sword_girls, cityscape, annoyed, running_on_liquid, outdoors, short_sleeves, supergirl, teardrop, girly_running, after_rain, short_shorts, blue_eyes, taking_shelter, city_lights, |_|, running_towards_viewer, nipple_tag, wet_floor, showering, mourning, ^^^, wet_jacket, w, chasing
```

debug_info：

```
匹配标签                      | 综合分    | 语义分    | 来源       | 中文含义
----------------------------------------------------------------------------------------------------
[提示] 结果过多 (129个)，已截取前 80 个。
short_dress                  | 0.961  | 1.000  | 短裙         | 短裙
city                         | 0.948  | 1.000  | 城市         | 城市
streaming_tears              | 0.938  | 1.000  | 流泪         | 流泪
street                       | 0.938  | 1.000  | 街道         | 街道
white_serafuku               | 0.927  | 1.000  | 白色水手服      | 白色水手服
clothes                      | 0.918  | 1.000  | 衣服         | 衣服
furious                      | 0.914  | 1.000  | 愤怒         | 愤怒
fume                         | 0.911  | 1.000  | 愤怒         | 愤怒
rain                         | 0.872  | 0.907  | 大雨         | 雨
blue_background              | 0.872  | 0.892  | 蓝色         | 蓝色背景
tears                        | 0.868  | 0.881  | 流泪         | 眼泪
miniskirt                    | 0.860  | 0.869  | 短裙         | 迷你裙
white_sailor_collar          | 0.856  | 0.882  | 白色水手服      | 白色水手领
shorts_under_skirt           | 0.849  | 0.894  | 短裙         | 裙下短裤
dressing                     | 0.849  | 0.898  | 穿着         | 穿衣
serafuku                     | 0.842  | 0.847  | 白色水手服      | 水手服
costume                      | 0.841  | 0.900  | 衣服         | 服装
shorts_under_dress           | 0.835  | 0.894  | 短裙         | 裙下短裤
dress                        | 0.829  | 0.816  | 穿着         | 连衣裙
crying                       | 0.828  | 0.845  | 流泪         | 哭泣
multiple_expressions         | 0.828  | 0.869  | 表情         | 多种表情
blue_skin                    | 0.825  | 0.854  | 蓝色         | 蓝色皮肤
undressing                   | 0.823  | 0.844  | 穿着         | 脱衣服
object_in_clothes            | 0.818  | 0.891  | 衣服         | 衣服里的物体
microskirt                   | 0.818  | 0.849  | 短裙         | 超短裙
tearing_up                   | 0.818  | 0.839  | 流泪         | 含泪
blue_theme                   | 0.815  | 0.842  | 蓝色         | 蓝色主题
emoji                        | 0.813  | 0.862  | 表情         | 表情符号
black_serafuku               | 0.812  | 0.838  | 白色水手服      | 黑色水手服
running                      | 0.810  | 0.835  | 奔跑         | 跑步
blue_shoes                   | 0.807  | 0.835  | 蓝色         | 蓝色鞋子
expression_chart             | 0.804  | 0.870  | 表情         | 表情表
emoticon                     | 0.800  | 0.862  | 表情         | 表情符号
facial_expression_training   | 0.796  | 0.874  | 表情         | 表情练习
>:(                          | 0.792  | 0.833  | 愤怒         | 愤怒面孔
shared_clothes               | 0.791  | 0.838  | 穿着         | 共穿衣服
clothes_on_and_off           | 0.787  | 0.855  | 衣服         | 穿脱中的衣服
blue_liquid                  | 0.784  | 0.865  | 蓝色         | 蓝色液体
jogging                      | 0.779  | 0.851  | 奔跑         | 慢跑
clothes_in_front             | 0.779  | 0.844  | 穿着         | 身前的衣服
wiping_tears                 | 0.776  | 0.825  | 流泪         | 擦眼泪
grey_serafuku                | 0.775  | 0.826  | 白色水手服      | 灰色水手服
urban                        | 0.769  | 0.825  | 城市         | 都市
angry                        | 0.765  | 0.778  | 愤怒         | 生气
shorts                       | 0.763  | 0.750  | 短裙         | 短裤
fleeing                      | 0.761  | 0.803  | 奔跑         | 逃跑
town                         | 0.760  | 0.801  | 城市         | 城镇
wet_clothes                  | 0.760  | 0.772  | 湿透         | 湿衣服
la_grondement_du_haine       | 0.758  | 0.815  | 愤怒         | 咆哮吧，我的愤怒
wet_face                     | 0.758  | 0.815  | 湿透         | 湿透的脸
slug_girl                    | 0.757  | 0.820  | 少女         | 蛞蝓少女
changing_clothes             | 0.754  | 0.816  | 穿着         | 换衣服
clothes_on_floor             | 0.754  | 0.809  | 衣服         | 地上的衣服
smile                        | 0.753  | 0.718  | 表情         | 微笑
angry_sex                    | 0.753  | 0.825  | 愤怒         | 愤怒性爱
wet_shirt                    | 0.750  | 0.772  | 湿透         | 湿透的衬衫
training_bra                 | 0.746  | 0.794  | 少女         | 少女内衣
wet_hair                     | 0.743  | 0.764  | 湿透         | 湿发
sword_girls                  | 0.742  | 0.787  | 少女         | 剑之少女
cityscape                    | 0.742  | 0.762  | 城市         | 城市景观
annoyed                      | 0.742  | 0.764  | 愤怒         | 恼怒
running_on_liquid            | 0.741  | 0.816  | 奔跑         | 在液体上奔跑
outdoors                     | 0.736  | 0.716  | 大雨         | 户外
short_sleeves                | 0.734  | 0.712  | 短裙         | 短袖
supergirl                    | 0.731  | 0.788  | 少女         | 超级少女
teardrop                     | 0.730  | 0.756  | 流泪         | 泪滴
girly_running                | 0.729  | 0.801  | 少女         | 少女跑
after_rain                   | 0.728  | 0.790  | 大雨         | 雨后
short_shorts                 | 0.725  | 0.718  | 短裙         | 超短裤
blue_eyes                    | 0.725  | 0.690  | 蓝色         | 蓝眼睛
taking_shelter               | 0.719  | 0.774  | 大雨         | 避雨
city_lights                  | 0.719  | 0.749  | 城市         | 城市灯光
running_towards_viewer       | 0.713  | 0.773  | 奔跑         | 奔向镜头
facial                       | 0.709  | 0.713  | 表情         | 面部
wet_floor                    | 0.701  | 0.750  | 大雨         | 潮湿的地面
showering                    | 0.701  | 0.727  | 大雨         | 洗澡
mourning                     | 0.699  | 0.762  | 流泪         | 哀悼
laughing                     | 0.695  | 0.707  | 表情         | 大笑
wet_jacket                   | 0.694  | 0.757  | 湿透         | 湿外套
chasing                      | 0.691  | 0.723  | 奔跑         | 追逐
```



### 关键词精准查找

你或许对某个词有模糊的印象，但不知道它对应的danbooru标签具体是什么。此时，你可以使用此节点精确查找。

**示例输入**

```
假肢
```

**建议参数**

- `top_k`：较小，例如5

- `结果上限`：较小，例如10

- `热度权重`：0.15

**示例输出**

tags_string：

```
running_blades, prosthetic_leg, peg_leg, severed_limb, fake_claws, fake_nails, detached_legs, multiple_legs, separated_legs, alternate_footwear
```

debug_info：

```
匹配标签                      | 综合分    | 语义分    | 来源       | 中文含义
----------------------------------------------------------------------------------------------------
running_blades               | 0.803  | 0.890  | 假肢         | 跑步假肢
prosthetic_leg               | 0.799  | 0.850  | 假肢         | 假腿
peg_leg                      | 0.777  | 0.850  | 假肢         | 假腿
severed_limb                 | 0.735  | 0.784  | 假肢         | 断肢
fake_claws                   | 0.710  | 0.760  | 假肢         | 假爪
fake_nails                   | 0.665  | 0.699  | 假肢         | 假指甲
detached_legs                | 0.657  | 0.714  | 假肢         | 断腿
multiple_legs                | 0.653  | 0.686  | 假肢         | 多条腿
separated_legs               | 0.653  | 0.696  | 假肢         | 叉开腿
alternate_footwear           | 0.640  | 0.683  | 假肢         | 替换鞋类
```



### 概念模糊查找

你可能想搜索关于某一个概念的关键词。此时，你可以使用此节点进行概念模糊查找。

**示例输入**

```
裙子
```

**建议参数**

- `top_k`：较大，例如40

- `结果上限`：较大，例如80

- `热度权重`：0.15

**示例输出**

tags_string：

```
skirt, dress, one_piece, white_skirt, black_skirt, upskirt, skirt_set, long_skirt, long_dress, pink_skirt, skirt_hold, skirt_lift, sweater_dress, skirt_suit, multicolored_skirt, skirt_pull, under_skirt, gown, dress_lift, unworn_skirt, skirt_tug, print_skirt, dress_tug, dress_flower, orange_skirt, flower_skirt, flower_dress, wet_dress, skirt_under_dress, wet_skirt, dress_ribbon, taut_skirt, armored_skirt, culottes, swimsuit_skirt, skort, skirt_rolled_up, dress_flip, skousers, dress_over_shirt, striped_clothes, dress_shirt, sleeveless_dress, undressing, clothes, lingerie, striped_dress, layered_dress, adjusting_clothes, multicolored_dress, clothes_between_thighs, hooded_dress, dressing, layered_clothes, clothes_on_floor, evening_gown, dress_shoes, hand_under_clothes, tearing_clothes, multicolored_clothes, clothed_female_nude_female, clothes_down, ballerina, holding_cloth, changing_clothes, shorts_under_dress, cloth, shirt_under_dress, clothes_on_and_off, folded_clothes, hand_under_dress, clothes_on_shoulders, dress_suit, reflective_clothes, night_clothes
```

debug_info：

```
匹配标签                      | 综合分    | 语义分    | 来源       | 中文含义
----------------------------------------------------------------------------------------------------
skirt                        | 0.987  | 1.000  | 裙子         | 裙子
dress                        | 0.869  | 0.862  | 裙子         | 连衣裙
one_piece                    | 0.835  | 0.862  | 裙子         | 连衣裙
white_skirt                  | 0.827  | 0.843  | 裙子         | 白裙子
black_skirt                  | 0.825  | 0.827  | 裙子         | 黑色裙子
upskirt                      | 0.824  | 0.852  | 裙子         | 裙底
skirt_set                    | 0.820  | 0.843  | 裙子         | 裙子套装
long_skirt                   | 0.820  | 0.844  | 裙子         | 长裙
long_dress                   | 0.819  | 0.844  | 裙子         | 长裙
pink_skirt                   | 0.814  | 0.835  | 裙子         | 粉色裙子
skirt_hold                   | 0.811  | 0.839  | 裙子         | 提裙子
skirt_lift                   | 0.805  | 0.820  | 裙子         | 掀起裙子
sweater_dress                | 0.804  | 0.841  | 裙子         | 毛衣裙
skirt_suit                   | 0.804  | 0.850  | 裙子         | 裙装西服
multicolored_skirt           | 0.803  | 0.845  | 裙子         | 多色裙子
skirt_pull                   | 0.803  | 0.841  | 裙子         | 拉裙子
under_skirt                  | 0.801  | 0.877  | 裙子         | 裙下
gown                         | 0.799  | 0.844  | 裙子         | 长裙
dress_lift                   | 0.797  | 0.820  | 裙子         | 掀起裙子
unworn_skirt                 | 0.796  | 0.834  | 裙子         | 未穿的裙子
skirt_tug                    | 0.793  | 0.841  | 裙子         | 拉裙子
print_skirt                  | 0.789  | 0.826  | 裙子         | 印花裙子
dress_tug                    | 0.788  | 0.839  | 裙子         | 提裙子
dress_flower                 | 0.786  | 0.827  | 裙子         | 裙装上的花
orange_skirt                 | 0.784  | 0.816  | 裙子         | 橙色裙子
flower_skirt                 | 0.781  | 0.863  | 裙子         | 花裙子
flower_dress                 | 0.781  | 0.863  | 裙子         | 花裙子
wet_dress                    | 0.779  | 0.833  | 裙子         | 湿裙子
skirt_under_dress            | 0.774  | 0.826  | 裙子         | 裙中裙
wet_skirt                    | 0.773  | 0.833  | 裙子         | 湿裙子
dress_ribbon                 | 0.767  | 0.822  | 裙子         | 裙子缎带
taut_skirt                   | 0.766  | 0.837  | 裙子         | 绷紧的裙子
armored_skirt                | 0.761  | 0.819  | 裙子         | 装甲裙
culottes                     | 0.761  | 0.830  | 裙子         | 裙裤
swimsuit_skirt               | 0.757  | 0.818  | 裙子         | 泳衣裙
skort                        | 0.755  | 0.830  | 裙子         | 裙裤
skirt_rolled_up              | 0.753  | 0.825  | 裙子         | 卷起的裙子
dress_flip                   | 0.750  | 0.820  | 裙子         | 掀裙子
skousers                     | 0.750  | 0.830  | 裙子         | 裙裤
dress_over_shirt             | 0.748  | 0.820  | 裙子         | 裙子穿在衬衫外
striped_clothes              | 0.706  | 0.689  | 裙子         | 条纹衣服
dress_shirt                  | 0.697  | 0.687  | 裙子         | 正装衬衫
sleeveless_dress             | 0.691  | 0.677  | 裙子         | 无袖连衣裙
undressing                   | 0.690  | 0.688  | 裙子         | 脱衣服
clothes                      | 0.688  | 0.729  | 裙子         | 衣服
lingerie                     | 0.686  | 0.684  | 裙子         | 女式内衣
striped_dress                | 0.673  | 0.682  | 裙子         | 条纹连衣裙
layered_dress                | 0.671  | 0.679  | 裙子         | 多层连衣裙
adjusting_clothes            | 0.669  | 0.672  | 裙子         | 整理衣服
multicolored_dress           | 0.668  | 0.680  | 裙子         | 多色连衣裙
clothes_between_thighs       | 0.667  | 0.701  | 裙子         | 腿间夹衣服
hooded_dress                 | 0.666  | 0.700  | 裙子         | 连帽裙
dressing                     | 0.664  | 0.681  | 裙子         | 穿衣
layered_clothes              | 0.662  | 0.679  | 裙子         | 层叠穿搭
clothes_on_floor             | 0.662  | 0.701  | 裙子         | 地上的衣服
evening_gown                 | 0.661  | 0.680  | 裙子         | 晚礼服
dress_shoes                  | 0.658  | 0.686  | 裙子         | 皮鞋
hand_under_clothes           | 0.657  | 0.676  | 裙子         | 手伸进衣服下
tearing_clothes              | 0.656  | 0.689  | 裙子         | 撕裂衣服
multicolored_clothes         | 0.655  | 0.674  | 裙子         | 彩色服饰
clothed_female_nude_female   | 0.655  | 0.677  | 裙子         | 着装女性与裸体女性
clothes_down                 | 0.654  | 0.677  | 裙子         | 衣服脱下一半
ballerina                    | 0.652  | 0.679  | 裙子         | 芭蕾舞者
holding_cloth                | 0.650  | 0.691  | 裙子         | 拿着布料
changing_clothes             | 0.649  | 0.692  | 裙子         | 换衣服
shorts_under_dress           | 0.647  | 0.674  | 裙子         | 裙下短裤
cloth                        | 0.647  | 0.685  | 裙子         | 布料
shirt_under_dress            | 0.642  | 0.672  | 裙子         | 裙下穿衬衫
clothes_on_and_off           | 0.641  | 0.683  | 裙子         | 穿脱中的衣服
folded_clothes               | 0.636  | 0.672  | 裙子         | 折叠好的衣服
hand_under_dress             | 0.627  | 0.686  | 裙子         | 手伸进裙底
clothes_on_shoulders         | 0.626  | 0.683  | 裙子         | 披在肩上的衣服
dress_suit                   | 0.625  | 0.679  | 裙子         | 正装
reflective_clothes           | 0.625  | 0.674  | 裙子         | 反光衣服
night_clothes                | 0.621  | 0.676  | 裙子         | 睡衣
```

## 使用建议

这个程序本来更适合[网页版](https://huggingface.co/spaces/SAkizuki/DanbooruSearch)使用。如果直接将此节点的输出链接到CLIP编码层，结果可能并不会很理想。我将其写成Comfy UI节点，是为了和我之前写过的一个插件[ComfyUI-NewBie-LLM-Formatter](https://github.com/SuzumiyaAkizuki/ComfyUI-NewBie-LLM-Formatter)联动。

这个插件的功能是利用LLM将用户的输入转换为`xml`格式的prompt供NewBie模型使用。但是，LLM经常会编造一些Danbooru上没有的标签。举个例子，如果用户输入了「水手服」，LLM可能会将其翻译为`sailor suit`，而「水手服」在Danbooru上真正的标签是`serafuku`。

为了解决这个问题，我希望可以给LLM一些标签做参考，让它在我给的标签集中选择，减少它编造标签的可能性。你可以像这样使用此节点：

![image-20260209220506092](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202602092205481.png)

然后，把这个子图输出的字符串连接到LLM XML Prompt Formatter的输入上

![image-20260209220730424](https://akizukipic.oss-cn-beijing.aliyuncs.com/img/202602092207684.png)

