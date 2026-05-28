# InfoSeek 与 MRAG-Bench 集成方案

## 现象层：数据格式差异

### MRAG-Bench 特征
```
{
  "id": "xxx",
  "question": "What is in the image?",  ← 文本问题
  "prompt": "...\nChoices:\nA: ...\nB: ...\nC: ...\nD: ...",  ← 多选题
  "answer": "A",  ← 单字母答案
  "gt_choice": "A",
  "query_image": <PIL Image>,
  "scenario": "outdoor", 
  "aspect": "color"
}
```

### InfoSeek 特征
```
Entity/Human:
{
  "data_id": "infoseek_test_00000000",
  "image_id": "oven_05494604",  ← 对应 images/all/oven_05494604.jpg
  "question": "What place inflows lake?"  ← 开放问题（无预定义选项）
}

Query:
{
  "data_id": "infoseek_train_00653800",
  "entity_id": "Q243381",
  "entity_text": "Macrolepiota procera"  ← entity 背景知识（无 image_id）
}
```

**核心差异**:
- ✅ MRAG: 多选题格式，4 个预定义选项，结构化评分
- ❌ InfoSeek: 开放式问题，无预定义选项，需要语义评分

---

## 本质层：三种集成方案

### 方案 A：转换为多选题（推荐 - 复用现有评估框架）

**目标**: 将 InfoSeek 开放问题转换成 MRAG-Bench 兼容的多选题

**流程**:
```
InfoSeek Entity/Human Record
    ↓
  提取: image_id, question
    ↓
  生成 4 个候选答案（通过 LLM / 知识库）
    ↓
  A: correct_answer
  B/C/D: plausible_distractors
    ↓
  转换为 MRAG-Bench 格式
    ↓
  调用现有 benchmark_XXX.py 评估
    ↓
  输出结果到 results/infoseek_mrag_format.jsonl
```

**优点**:
- 完全复用 LLaVA/MagicLens/RRF 评估框架
- 结果可与 MRAG-Bench 直接对比
- 评分逻辑一致（4 选 1）

**缺点**:
- 需要生成高质量的干扰项
- 可能引入额外的 LLM 调用成本
- 原始 InfoSeek 任务的难度特征可能改变

---

### 方案 B：开放式评估（原生支持 - 新评估流程）

**目标**: 保持 InfoSeek 的开放问题性质，用语义匹配评分

**流程**:
```
InfoSeek Record
    ↓
  加载图片 + 问题
    ↓
  LLaVA / MagicLens 检索 + 生成答案
    ↓
  答案: "A place called the River Valley"
    ↓
  语义相似度评分
    (BERT/GPT / 精确词汇匹配)
    ↓
  输出结果到 results/infoseek_open_format.jsonl
```

**评分策略**:
```
1. 精确匹配 (Exact Match)
   - 答案包含真值关键词 → 得分 1.0

2. 部分匹配 (Partial Match)
   - 词汇重叠率 > 50% → 得分 0.5

3. 语义相似度 (Semantic)
   - 使用 SBERT / E5 embeddings
   - 相似度 > threshold → 得分

4. LLM 判断 (GPT-as-Judge)
   - 用 GPT/Claude 判断答案是否正确
```

**优点**:
- 保持原始任务复杂性
- 更接近真实场景（开放问答）
- 可评估系统的生成能力（非仅检索排名）

**缺点**:
- 评分标准难以标准化
- 与 MRAG-Bench 结果不可直接对比
- 需要实现新的评估框架

---

### 方案 C：作为图像语料库扩展（最小成本）

**目标**: 将 InfoSeek 图像混入 MRAG-Bench corpus，测试检索效果

**流程**:
```
MRAG-Bench 问题集
    ↓
  (保持原样，使用 MRAG-Bench 的 test split)
    ↓
  扩展 corpus_dir:
  corpus/
  ├── mrag_corpus/          (原 MRAG 图像库)
  └── infoseek_images/all/  (InfoSeek 100万+ 图像)
    ↓
  运行 benchmark_corpus_rag.py --corpus-dir corpus/
    ↓
  对比：
  - 只用 MRAG corpus
  - 混合 MRAG + InfoSeek corpus
  - 只用 InfoSeek corpus
```

**优点**:
- 最小改动，复用现有代码
- 评估 corpus 规模对检索效果的影响
- 可快速验证 InfoSeek 图像质量

**缺点**:
- 无法评估 InfoSeek 数据本身的难度
- 结果混淆了"数据集特性"和"图像库规模"的影响

---

## 哲学层：选择建议

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| **对标 MRAG-Bench** | 方案 A | 统一评分标准，可直接对比 |
| **评估系统的生成能力** | 方案 B | 开放问题更接近真实应用 |
| **快速验证效果** | 方案 C | 最小成本，最快反馈 |
| **全面研究** | A + B + C | 三个维度完整评估 |

---

## 架构决策

### **推荐：方案 A + C 并行**

**理由**:
1. 方案 A：生成 infoseek_as_mrag.py，转换为多选题，与 MRAG 对标
2. 方案 C：symlink 或配置 corpus_dir，扩展检索语料库
3. 两条线同时进行，成本小，收获大

**代码结构**:
```
src/mrag/
├── infoseek_loader.py          ← 方案 A/B 的数据加载适配层
└── infoseek_converter.py       ← 方案 A 的多选题转换器

test/
├── benchmark_infoseek.py       ← 方案 A：InfoSeek 多选题评估
├── benchmark_infoseek_open.py  ← 方案 B：开放式评估（可选）
└── benchmark_corpus_rag.py     ← 方案 C：直接复用

results/
├── infoseek_mrag_format/       ← 方案 A 输出
├── infoseek_open_format/       ← 方案 B 输出（可选）
└── corpus_comparison/          ← 方案 C 对比报告
```

---

## 下一步行动

### 立即开始（第 1 阶段 - 1-2 天）

**方案 A 实现**:
1. ✅ `src/mrag/infoseek_loader.py` - JSONL 迭代器 + 图像加载
2. ✅ `src/mrag/infoseek_converter.py` - LLM 调用生成多选项
3. ✅ `test/benchmark_infoseek.py` - 复用 LLaVA/MagicLens 评估
4. ✅ 适配 `github/MRAG-Bench/eval/score.py` 评分

**方案 C 验证**:
1. 确认 corpus_dir 配置支持多路径
2. 跑单组对比实验验证可行性

### 后续扩展（第 2 阶段 - 可选）

**方案 B**:
- 实现语义评分器
- 集成 LLM-as-judge 流程

**分析**:
- InfoSeek 难度分析
- 与 MRAG-Bench 的 scenario/aspect 特性对比

---

## 数据关联与部署约束

### 本地数据状态

```bash
# ✅ 元数据已本地化
wc -l data/infoseek/Entity/*.jsonl  # 1,001,028 条记录
wc -l data/infoseek/Human/*.jsonl  # 8,884 条记录
wc -l data/infoseek/Query/*.jsonl  # 660,000+ 条

# ❌ 图像文件未本地化（按设计 - 57GB 太大）
ls data/infoseek/images/all  # 空目录

# 图像实际位置
# 本地: /mnt/d/mRAG/data/infoseek/images/all/ （空，待配置）
# 远程: /home/user/code/mRAG/data/infoseek/images/all/ （1,005,415 张，57GB）
```

### 因此：运行环境限制

| 组件 | 本地开发 | 远程评估 |
|------|---------|---------|
| 数据加载器 | ✅ 可测试（仅验证元数据解析） | ✅ 可完整运行 |
| 多选题转换 | ✅ 可模拟（LLM 生成选项，不需图片） | ✅ 可完整运行 |
| 图像检索 + 答案生成 | ⚠️ 需要图片（拉取子集 OR NFS 挂载） | ✅ 可完整运行 |

### 解决方案 A：本地使用子集数据

```bash
# 从服务器拉取前 100 条记录对应的图片（~10-20MB）
# 1. 提取 test split 的前 100 个 image_id
head -100 data/infoseek/Entity/infoseek_test.jsonl | \
  python3 -c "
    import sys, json
    for line in sys.stdin:
      print(json.loads(line)['image_id'])
  " > /tmp/image_ids_sample.txt

# 2. 从服务器指定拉取这些图片
# （需要编写 rsync filter 脚本）
```

### 解决方案 B：远程评估 + 结果回传

```bash
# 推荐方案：
# 1. benchmark_infoseek.py 在 nnu 上执行
# 2. 结果 JSONL 回传本地
# 3. 本地运行评分脚本
```

---

## GEB PROTOCOL - 架构决策记录

> **决策**: 选择 方案 A + C 并行路线

**变更触发**:
- 新增 2 个模块：infoseek_loader.py, infoseek_converter.py
- 新增 1 个评估脚本：benchmark_infoseek.py
- 修改：corpus_dir 配置支持多路径

**下步检查清单**:
- [ ] L2: 更新 `src/mrag/CLAUDE.md` 添加 infoseek_loader 职责
- [ ] L2: 更新 `test/CLAUDE.md` 添加 benchmark_infoseek.py 职责
- [ ] L3: 为新文件添加头部注释 [INPUT]/[OUTPUT]/[POS]

**验收标准**:
- ✅ benchmark_infoseek.py 能运行 10 条 smoke test
- ✅ 结果格式与 MRAG-Bench score.py 兼容
- ✅ 与 MRAG-Bench 基准的差异在 documentation 中清晰说明

