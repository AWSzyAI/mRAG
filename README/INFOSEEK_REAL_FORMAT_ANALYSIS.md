# InfoSeek 真实数据格式分析 & 策略重新评估

## 🔍 关键发现

**InfoSeek 实际数据格式**:
```json
// Entity split (347K 样本) - 仅包含
{
  "data_id": "infoseek_test_00000000",
  "image_id": "oven_05494604",
  "question": "What place inflows lake?"
  // ❌ 没有 options
  // ❌ 没有 correct (GT 答案)
}

// Human split (8.9K 样本) - 同样格式
// Query split (1.0M 样本) - 完全不同格式，不含图像

// MRAG-Bench 需要的格式:
{
  "data_id": "...",
  "image_id": "...",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, // ✅ 需要
  "correct": "A"  // ✅ 需要
}
```

**结论**: InfoSeek Entity/Human split 本质上是**开放式 VQA**，不是多选题

---

## 📊 这改变了什么？

### 原始分析
"能不能仅用 Entity split (347K)？" → 不能，因为以为它是多选题

### 修正后的分析
"能不能仅用 Entity split？" → **可以！**，但要改变评估方式

---

## 🎯 现在的真实选择

### 选项 1: 转换 → MRAG-Bench 兼容 (当前实现)

```
InfoSeek 开放式问题
    ↓
LLM 生成 {A, B, C, D} + correct
    ↓
MRAG-Bench 框架
```

**这是必要的吗？**
- ✅ 如果你想复用 MRAG-Bench 代码和基线对标
- ✅ 如果你想用 MagicLens + LLaVA（它们本来就是为多选题设计的）
- ❌ 如果你只想评估 InfoSeek 本身

**成本**: ~$500 (LLM API) + 人力 (审核)

---

### 选项 2: 保持开放式，独立评估框架 (推荐用原始格式!)

```
InfoSeek 开放式问题 (原始格式)
    ↓
模型生成: "river outlet"
    ↓
评分:
  - 方式 A: 语义相似度 (BERTScore) ← 推荐
  - 方式 B: LLM 判断 (GPT-4) 
  - 方式 C: 人工评估
    ↓
结果: 开放式 VQA 准确率
```

**这更好吗？**
- ✅ **保留原始数据特性** (不改造数据)
- ✅ 更贴近真实 VQA 任务
- ✅ 无 LLM 生成干扰项的风险
- ✅ 成本低 ($0-30)
- ✅ **完全独立于 MRAG-Bench**
- ❌ 评分标准需要清晰定义
- ❌ 难以与 MRAG-Bench 基线对标

**我的建议**: **这条路更合适**！

---

## 💡 比较总结

| 维度 | 转换方案 | 开放式方案 |
|------|---------|---------|
| **数据改造** | ❌ 大改 (生成干扰项) | ✅ 无改 (原样使用) |
| **格式** | 四选一 | 开放式 |
| **GT 质量** | 🟡 LLM 生成 | ✅ 保留原始 |
| **评估指标** | MRAG 准确率 | 语义相似度 |
| **与 MRAG 对标** | ✅ 容易 | ❌ 困难 |
| **独立性** | ❌ 依赖框架 | ✅ 完全独立 |
| **科研价值** | 中 | **⭐⭐⭐ 高** |
| **代码复用** | ✅ MagicLens/LLaVA | ❌ 需要改写 |
| **成本** | 高 | 低 |

---

## 🎓 推荐方案: 开放式 VQA 评估

### 第一步: 选择评分方法

#### 方法 A: BERTScore (推荐 ⭐⭐⭐)

```python
from bert_score import score

# 模型输出
predictions = ["river outlet", "place where river enters lake"]

# GT 答案 (可有多个)
references = [["lake outlet"], ["lake outlet"]]

# 计算得分
precision, recall, f1 = score(predictions, references, lang="en")
# → 返回向量化分数，直观可用
```

**优点**:
- 免费 (本地计算)
- 语义感知 (相比 BLEU)
- 支持多个参考答案
- 快速 (~100ms/样本)

**缺点**:
- 不完全不同的答案格式仍然可能得分 (需要多参考)

#### 方法 B: LLM 判断 (更灵活但更贵)

```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": f"""
判断以下答案是否正确:

问题: {question}
图像: [image visible to model]
预测答案: {prediction}

回答: "是" 或 "否"，并给出置信度 (0-1)
"""
    }]
)
```

**优点**:
- 最灵活，考虑图像上下文
- 处理多个合理答案

**缺点**:
- 成本 $0.01-0.03/样本
- 1.3M 样本 → $13,000-39,000
- 不可完全复现

---

### 第二步: 构建评估框架

```python
# 1. 加载 InfoSeek 开放式问题
questions = load_infoseek_entity_split()

# 2. 为子集生成 GT (方式选择)
# - 选项 A: 用 LLM 生成可靠答案 (仅用 1-2 次)
# - 选项 B: 用众包标注 (最可靠)
# - 选项 C: 用文本匹配找候选 (成本低)
gt_answers = generate_gt_answers(questions[:1000])

# 3. 运行模型
model_outputs = model_predict(questions)

# 4. 评分
scores = score_with_bertscore(model_outputs, gt_answers)

# 5. 结果
print(f"F1 Score: {scores['f1'].mean():.3f}")
print(f"准确率 (完全匹配): {exact_match_rate:.1%}")
```

---

### 第三步: 生成 GT 答案 (关键步骤)

三个选项，按推荐度:

#### 选项 1: 众包标注 (最好，但成本高) ⭐⭐⭐

```
招募 5-10 个标注员
每个问题 3 个标注
投票多数通过 → GT 答案
Cohen's Kappa ≥ 0.8 验证一致性
```

成本: $5,000-10,000 (200-500 样本)

#### 选项 2: 单一高质量 LLM (折中) ⭐⭐

```python
converter = InfoSeekConverter(
    llm_model="gpt-4",  # 更好的模型
    cache_db="...",
    # 只生成答案，不生成干扰项
)

# 改进 prompt，明确指示"只给一个答案"
results = []
for q in questions[:1000]:
    answer = converter._call_llm(q)["answer"]
    results.append(answer)
```

成本: $3,000-5,000 (1000 样本用 gpt-4)

#### 选项 3: 文本检索 + LLM (最低成本) ⭐

```
使用维基百科、维基数据等知识库
搜索问题相关的概念答案
手工验证 top-100
```

成本: $0 (仅人力审核)

---

## ✅ 最终建议

### 立即行动 (本周)

1. **确认开放式评分框架**
   ```python
   # 安装 BERTScore
   pip install bert-score
   
   # 快速测试 5 个样本
   from bert_score import score
   predictions = ["answer1", "answer2"]
   references = [["gt1"], ["gt2"]]
   precision, recall, f1 = score(predictions, references, lang="en")
   ```

2. **为 100 个样本手工标注 GT**
   - 随机选 100 个 Entity test 样本
   - 3 个人各标注一遍
   - 投票通过
   - 成本: 15 小时 × $50/hr = $750

3. **建立基准线**
   ```python
   # 用几个简单的模型跑 100 个样本
   # 测试 BERTScore 的信号强度
   baseline_f1 = measure_baseline()
   ```

### 后续选择

**如果 BERTScore 信号好** (F1 0.3+):
- ✅ 继续扩展到 1000+ 样本
- ✅ 不用转换为多选题
- ✅ 保持 InfoSeek 原始特性
- ✅ **独立出论文: "InfoSeek: Large-Scale Open-Ended VQA Dataset"**

**如果 BERTScore 信号不好**:
- 考虑回到多选一转换方案 (方案 A)
- 或者用 LLM 判断替代 BERTScore

---

## 🎓 为什么这个方向更好？

1. **保留原始数据** - 不改造，直接用
2. **科研创新** - 开放式评分是新方向，比多选题评估更有趣
3. **无 GT 依赖** - 不用依赖 LLM 生成的伪答案
4. **成本低** - BERTScore 免费，即使用 LLM 也只需 1000 个
5. **可独立** - 完全不需要 MRAG-Bench 或 MagicLens
6. **更实用** - 真实 VQA 应用中就是开放式答案

---

## 📋 行动清单

- [ ] 安装 BERTScore
- [ ] 手工标注 100 个样本 GT
- [ ] 跑基准线测试
- [ ] 如果信号好，启动 1000+ 样本标注
- [ ] 决定是否需要转换为多选题 (取决于结果)

---

**核心转变**: 不是「如何改造 InfoSeek 适配 MRAG-Bench」，而是「如何评估 InfoSeek 的原始特性」！
