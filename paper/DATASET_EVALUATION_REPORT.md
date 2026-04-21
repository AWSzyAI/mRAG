# 数据集评估与选型报告

## 一、当前数据集：MRAG-Bench

### 1. 数据集概览

| 属性 | 值 |
|------|-----|
| **数据集名称** | MRAG-Bench (Multi-modality RAG Benchmark) |
| **来源** | HuggingFace: `uclanlp/MRAG-Bench` |
| **样本总数** | 1353 个 QA 对 |
| **图像语料库** | COCO 2017 + ImageNet + Fine-grained Dataset |
| **图像总数** | ~30K+ 张（多源混合） |
| **问题类型** | 单跳推理为主 + 多属性查询 |
| **评估方式** | 精确匹配 (EM) + 模糊匹配 |
| **任务定义** | VQA (Visual Question Answering) with Retrieval |

### 2. 场景分布（9 个场景）

| 场景 | 样本数 | 特征 | 难度 | MagicLens 表现 |
|------|--------|------|------|--------|
| **Scope** | 102 | 搜索范围缩小、背景变化 | 中 | 55.88% ✅ |
| **Deformation** | 103 | 物体变形、形状改变 | 中 | 57.26% ✅ |
| **Others** | 30 | 其他杂项 | 中 | 56.67% ✅ |
| **Angle** | 102 | 视角/角度变化 | 中 | 53.92% ✅ |
| **Obstruction** | 102 | 遮挡、遮蔽 | 中 | 51.96% ✅ |
| **Biological** | 102 | 生物识别、分类 | 中 | 48.04% ❌ |
| **Partial** | 102 | 局部/截断 | 中 | 49.02% ❌ |
| **Temporal** | 102 | 时间变化、状态演变 | 难 | 46.08% ❌ |
| **Incomplete** | 102 | 信息不完整 | 难 | 26.47% ❌❌ |

### 3. 数据质量评估

| 维度 | 评分 | 说明 |
|------|------|------|
| **覆盖度** | ⭐⭐⭐⭐ | 9 个场景全面覆盖变化类型 |
| **多样性** | ⭐⭐⭐⭐ | 多源语料库（COCO+ImageNet+细粒度） |
| **难度分级** | ⭐⭐⭐ | 场景偏向中等难度，缺少极难样本 |
| **标注质量** | ⭐⭐⭐⭐ | 官方精心设计，参考图+GT 高度可信 |
| **评估一致性** | ⭐⭐⭐⭐ | 90.84% 协议一致度，验证结果稳定 |

### 4. 与本研究的匹配度

| 匹配项 | 符合 | 说明 |
|--------|------|------|
| query_image + question 输入 | ✅ | 核心输入格式一致 |
| 图像语料库检索 | ✅ | MRAG-Bench 本身就是图像检索为主 |
| LLaVA 生成验证 | ✅ | LLaVA 能准确回答选择题 |
| VQA 评估框架 | ✅ | 精确匹配、场景分布 |
| **总体匹配度** | **⭐⭐⭐⭐⭐** | 完美适配当前 pipeline |

---

## 二、候选第二数据集深度分析

### 候选 1️⃣：InfoSeek（Google Research）

#### 1.1 基本信息

| 属性 | 值 |
|------|-----|
| **论文** | InfoSeek: Fact-based Question Answering about Real-world Entities |
| **来源** | HuggingFace: `google/infoseek` |
| **样本总数** | 14,295 个 QA 对 |
| **图像数** | 22,500+ 张（多国维基百科） |
| **问题类型** | 实体知识查询、地理位置、历史事件 |
| **语言** | 英文 + 中文 |
| **发布时间** | 2023 年 |
| **引用量** | 高（Google 官方） |

#### 1.2 任务定义

**输入**：
```json
{
  "entity": "Paris",
  "question": "What is the capital of France?",
  "image": "paris_eiffel_tower.jpg",
  "image_context": "A famous landmark"
}
```

**输出**：
```json
{
  "answer": "Paris is the capital of France",
  "provenance": "Wikipedia + Image evidence"
}
```

#### 1.3 与 MagicLens 的兼容性

| 维度 | 兼容性 | 说明 |
|------|--------|------|
| **query_image** | ✅ 高 | 有实体关联图像 |
| **question** | ✅ 高 | 自然语言问题 |
| **图像语料** | ✅ 高 | 多国维基百科图像 |
| **检索任务** | ✅ 高 | 知识检索 + 视觉验证 |
| **精度评估** | ✅ 中等 | 用 LLaVA 评估 |
| **现有脚本适配** | 🟡 中等 | 需改评估逻辑（从选择题 → 开放式回答） |

#### 1.4 优势

✅ **知识密集型**：MagicLens 在属性检索上有优势
✅ **跨语言**：可评估多语言迁移
✅ **大规模**：14K 样本，足够消融
✅ **公开数据集**：有官方 baseline 可对标
✅ **与 MRAG-Bench 互补**：强调知识 vs 变化识别

#### 1.5 劣势

❌ 答案格式不同（选择题 → 开放式）
❌ 评估指标需定制（BLEU / ROUGE vs EM）
❌ 图像质量参差（维基百科 vs COCO）

#### 1.6 适配工作量

| 工作 | 工作量 | 时间 |
|------|--------|------|
| 下载与处理 | 🟢 小 | 1 天 |
| 语料库构建 | 🟡 中 | 2 天 |
| 评估脚本改写 | 🟡 中 | 1 天 |
| 消融验证 | 🟢 小 | 1 天 |
| **总计** | **🟡 中等** | **~5 天** |

#### 1.7 推荐度

**⭐⭐⭐⭐⭐ 强烈推荐**

理由：
- 与 MRAG-Bench 互补（知识 + 变化识别）
- 适配工作相对简单
- Google 官方数据集，权威性高
- 有跨语言潜力
- 暑假前能完成迁移验证

---

### 候选 2️⃣：GQA（Scene Graph Question Answering）

#### 2.1 基本信息

| 属性 | 值 |
|------|-----|
| **论文** | GQA: A New Dataset for Real-World Visual Reasoning |
| **来源** | HuggingFace: `dleemiller/gqa` |
| **样本总数** | 22M 个平衡 QA 对 |
| **图像数** | 1.2M 张（来自 Flickr + 合成） |
| **问题类型** | 场景理解、对象关系、空间推理 |
| **特点** | 场景图结构化标注 |
| **引用量** | 超高（顶级 CV 会议） |

#### 2.2 任务定义

**场景图示例**：
```
Image: Kitchen scene
Objects: [stove, pot, countertop]
Attributes: pot[color=red, size=large]
Relations: pot[on]stove, stove[in]kitchen
```

**问题类型**：
- 单跳：`What color is the pot?`
- 多跳：`Is the red pot on the stove?`
- 推理：`Are there more pots than pans?`

#### 2.3 与 MagicLens 的兼容性

| 维度 | 兼容性 | 说明 |
|------|--------|------|
| **query_image** | ✅ 高 | 完整场景图像 |
| **question** | ✅ 高 | 自然语言 |
| **图像语料** | ✅ 高 | 大规模 1.2M |
| **检索任务** | 🟡 中 | 需要构建 query-aware 子图检索 |
| **精度评估** | ✅ 高 | 直接选择题格式 |
| **现有脚本适配** | 🟡 中 | 可直接使用评估框架 |

#### 2.4 优势

✅ **超大规模**：22M 样本，消融统计力强
✅ **结构化**：场景图使问题定义精确
✅ **多跳推理**：比单跳更复杂
✅ **开源活跃**：社区讨论热烈

#### 2.5 劣势

❌ **数据过大**：22M 样本跑完需数周
❌ **与 MRAG-Bench 冗余**：都是视觉推理
❌ **难度跨度大**：可能掩盖 MagicLens 的真实表现
❌ **图像质量下降**：部分合成图可能不适合检索

#### 2.6 推荐度

**⭐⭐⭐ 可选**

理由：
- 暑假前难以完整评估（22M 太大）
- 与 MRAG-Bench 功能重叠
- 但作为**长期**验证数据集不错

---

### 候选 3️⃣：OK-VQA（Outside Knowledge VQA）

#### 3.1 基本信息

| 属性 | 值 |
|------|-----|
| **论文** | OK-VQA: A Visual Question Answering Benchmark |
| **来源** | HuggingFace: `okvqa` |
| **样本总数** | 14,055 个 QA 对 |
| **图像数** | 14,055 张（Microsoft COCO） |
| **特点** | 必须外部知识才能回答 |
| **评估方式** | 自由形式文本评分 |

#### 3.2 兼容性

| 维度 | 符合 |
|------|------|
| 图像检索 | ✅ 适配 |
| 知识融合 | 🟡 需集成知识库 |
| 评估框架 | 🟡 需改为文本评分 |

#### 3.3 推荐度

**⭐⭐ 低**

理由：
- 需要外部知识库集成（超出 MagicLens 范围）
- 样本数与 InfoSeek 相当但质量低
- 与 MRAG-Bench 冗余度高

---

### 候选 4️⃣：VCR 1.0（Visual Commonsense Reasoning）

#### 4.1 基本信息

| 属性 | 值 |
|------|-----|
| **论文** | From Recognition to Cognition: Visual Commonsense Reasoning |
| **样本数** | 290K QA 对 + 110K 图 |
| **特点** | 多跳推理、常识推理透明 |
| **答案格式** | 多项选择（A/B/C/D） |

#### 4.2 推荐度

**⭐⭐⭐ 可选**

理由：
- 与 MRAG-Bench 功能非常相似
- 推理难度更高（好处）
- 但迁移学习收益有限

---

## 三、最终推荐方案

### Phase 优先级

#### **Phase D（第 1-2 周）：MRAG-Bench 消融**
```
D01-D12 实验系列
确认多维度分解的有效性
目标精度：52-54%（超 CLIP 50.41%）
```

#### **Phase E（第 3-4 周）：InfoSeek 迁移**
```
E_InfoSeek:
  - 构建图像语料库
  - 适配评估脚本
  - 运行最优配置（D_best）
目标：验证跨数据集泛化能力
```

#### **Phase F（选项）：GQA 大规模验证**
```
F_GQA: 
  - 可行性测试（100K 样本）
  - 完整验证（如时间允许）
```

### 数据集选型总结表

| 数据集 | 样本数 | 推荐度 | 优先级 | 备注 |
|--------|--------|--------|--------|------|
| **MRAG-Bench** | 1.3K | ⭐⭐⭐⭐⭐ | 1️⃣ | 当前主数据集，必做 |
| **InfoSeek** | 14K | ⭐⭐⭐⭐⭐ | 2️⃣ | 推荐迁移，互补优势 |
| GQA | 22M | ⭐⭐⭐ | 3️⃣ | 可选大规模验证 |
| VCR 1.0 | 290K | ⭐⭐⭐ | 3️⃣ | 推理更难，可选 |
| OK-VQA | 14K | ⭐⭐ | ❌ | 不推荐（冗余） |

### 论文贡献主张

**当前**：
- ✅ 诊断了 MagicLens 直检索问题
- ✅ 提出多维度查询分解框架
- ⏳ 在 MRAG-Bench 上的改进验证

**暑假前目标**：
- ✅ Phase D 完成，得到最优配置（D_best）
- ✅ Phase E 完成，InfoSeek 迁移验证
- ⏳ 两数据集上的一致改进证实

**论文故事线**：
> 我们发现了多模态检索中的"语义原型偏置"问题，提出多维度查询分解框架有效缓解。在两个权威数据集（MRAG-Bench + InfoSeek）上验证了方法的有效性和泛化能力。

---

## 四、InfoSeek 适配工作清单

### 4.1 数据集准备（1 天）

```bash
# 下载数据集
huggingface-cli download google/infoseek --repo-type dataset

# 检查格式
python -c "
from datasets import load_dataset
ds = load_dataset('google/infoseek')
print(ds['train'][0])
"

# 输出应该包含：
# - entity, question, answer
# - positive_image_paths, negative_image_paths
```

### 4.2 图像语料库构建（2 天）

```bash
# 下载维基百科图像
python scripts/download_infoseek_images.py \
  --output-dir data/infoseek_images \
  --num-workers 8

# 构建 CLIP 索引
python -c "
import torch
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import numpy as np

# 扫描所有图像
images = sorted(Path('data/infoseek_images').glob('**/*.jpg'))
print(f'Found {len(images)} images')

# 构建嵌入索引
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').cuda()
# ... 批量编码 ...
"
```

### 4.3 评估脚本改写（1 天）

从 MRAG-Bench 改写到 InfoSeek：

```python
# 当前（MRAG-Bench）
answer = "A"  # 选择题
correct = answer == ground_truth

# 改写（InfoSeek）
answer = model.generate(...)  # 开放式回答
correct = compute_bleu(answer, ground_truth) > threshold
```

### 4.4 运行最优配置（1 天）

```bash
# 使用 Phase D 确定的最优超参数
# 例：3维 Qwen2.5-7B + RRF

DIM_GENERATOR_API_KEY="..." \
NUM_DIMENSIONS=3 \
FUSION_STRATEGY="rrf" \
python test/pipeline_multi_dim_rag.py \
  --dataset infoseek \
  --corpus-dir data/infoseek_images \
  --answers-file log/E_InfoSeek/results.jsonl
```

---

## 五、论文截止与投稿计划

| 事项 | 日期 | 状态 |
|------|------|------|
| Phase D01 启动 | 2026-04-16 | ✅ 今日 |
| Phase D 完成 | 2026-05-01 | 🟡 进行中 |
| InfoSeek 迁移开始 | 2026-05-05 | 🔴 待启动 |
| InfoSeek 结果完成 | 2026-05-20 | 🔴 待启动 |
| 论文初稿第一版 | 2026-05-25 | 🔴 待启动 |
| 导师反馈与修改 | 2026-06-10 | 🔴 待启动 |
| **投稿 IJCAI** | **2026-06-30** | 🔴 **目标** |

