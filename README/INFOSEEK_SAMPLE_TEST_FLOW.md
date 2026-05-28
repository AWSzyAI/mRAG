# InfoSeek 单题测试流程示例

目的：展示从单条 InfoSeek 记录启动的一次完整测试流程（本地可复现的最小步骤），分析是否存在 Ground Truth (GT)，并给出将该测试接入 MRAG-Bench 的具体方案与理由。

---

**选取样例（来自 `data/infoseek/Entity/infoseek_test.jsonl` 首行）**

```json
{
  "data_id": "infoseek_test_00000000",
  "image_id": "oven_05494604",
  "question": "What place inflows lake?"
}
```

翻译说明：`image_id` 对应图片文件名（例如 `oven_05494604.jpg`），图片位于远程：`/home/user/code/mRAG/data/infoseek/images/all/`。

---

## 一、单题测试流程（不依赖全部图片库）

目标：用一条记录跑通加载→（可选）检索→生成答案→评分的完整流程。

前提：本地没有图片时，使用远程挂载或仅做逻辑验证（跳过图像相关步骤）。

步骤：

1. 环境准备（本地开发/远程运行二选一）

- 本地快速验证（不需要图片）：

```bash
python3 test/test_infoseek_metadata.py   # 验证元数据可读
```

- 若需真实图像（混合/远程模式），可 NFS/sshfs 挂载远程图片目录：

```bash
# 例：sshfs
mkdir -p /mnt/infoseek_images
sshfs nnu:/home/user/code/mRAG/data/infoseek/images/all /mnt/infoseek_images
export INFOSEEK_IMAGE_DIR=/mnt/infoseek_images
```

2. 加载单条记录并映射到加载器接口

```python
from src.mrag.infoseek_loader import InfoSeekDataset

ds = InfoSeekDataset('/mnt/d/mRAG/data/infoseek')
# 只示例第一条
rec = next(ds.iter_split('entity_test', load_image=False))
print(rec.data_id, rec.image_id, rec.question)
```

3. 若有图片且需要视觉输入：加载图片（或确保挂载路径可访问）

```python
# 如果 INFOSEEK_IMAGE_DIR 已挂载到 ds.images_dir
rec_with_img = next(ds.iter_split('entity_test', load_image=True))
img = rec_with_img.query_image  # PIL.Image
```

4. 生成多选候选（若需要以 MRAG-Bench 多选题格式评估）

- 方式 A：人工 GT（优先）——请人工标注或查来源
- 方式 B：用 LLM 生成：提示模板生成 3 个干扰项 + 正确/候选项

示例（伪代码）：

```python
# 使用你们的 LLM 接口（示例伪代码）
question = rec.question
prompt = f"Generate 3 plausible but incorrect distractors for: {question}\nReturn JSON: {{'A': correct, 'B': ..., 'C': ..., 'D': ...}}"
choices = call_llm(prompt)
```

建议：对 LLM 生成的选项做简单质量过滤（长度、重复率、与原问题相关性）。

5. 构造 MRAG-Bench 风格的样本并保存为 JSONL

```jsonl
{"id": "infoseek_test_00000000", "question": "<question>\n Choices:\nA: ...\nB: ...\nC: ...\nD: ...", "answer": "A", "image_files": ["/mnt/infoseek_images/oven_05494604.jpg"]}
```

6. 调用 MRAG-Bench / test harness 运行一次推理（smoke test，N=1 或 N=10）

```bash
# 举例：使用已有 test/benchmark_corpus_rag.py 或 test/benchmark_magiclens.sh 的最小运行
MAX_SAMPLES=1 python3 test/benchmark_magiclens.py --dataset-name local/infoseek_mrag --corpus-dir /path/to/corpus
```

7. 得到模型输出并评分（若有 GT）或语义评估（若无 GT）

- 若已有人工 GT（选项中包含 `answer` 字段），直接用 `github/MRAG-Bench/eval/score.py` 打分。
- 若无 GT（InfoSeek 原始记录无标签），可用两种替代：
  - 人工评估（将模型答案与人工标签比对）
  - 使用 `LLM-as-judge`：把模型答案和候选 GT 交给高质量 LLM 判定（作为近似评估）

示例命令：

```bash
python3 github/MRAG-Bench/eval/score.py -i results/infoseek_mrag_results.jsonl
```

---

## 二、关于 Ground Truth (GT) 的分析

- InfoSeek `Entity` / `Human` 文件示例字段通常仅包含 `data_id`, `image_id`, `question`。
- 当前本地元数据检查显示 **没有明确的 GT 答案字段**（即没有 `answer` / `label` 字段）。
- `Human` 文件名看起来像人工评测集，但内容仍只有 `question` 与 `image_id`，因此并非直接可用的准确标签。

结论：

- 原生 InfoSeek 数据集**不包含可直接用于精确评分的 GT 标签**。要做分类式（多选）准确率评估，必须额外获取 GT：
  1. 从原始数据源/作者处请求标签；或
  2. 对一小批样本做人工标注（优先项）；或
  3. 使用 LLM 生成近似 GT 并通过人工审查（临时方案），或
  4. 采用开放式语义评分（参见方案 B）。

影响：若没有 GT，不能直接用 `score.py` 计算 Accuracy。可以采用语义相似度评分或 LLM judge，但这些应在实验报告中注明“评分方法不同”。

---

## 三、将 InfoSeek 接入 MRAG-Bench 的方案与理由（建议：方案 A 为主，方案 C 辅助）

### 方案 A（推荐主线）：将 InfoSeek 转为 MRAG-Bench 多选题

- 流程要点：
  1. 对所选样本（可批量）生成或采集 GT 答案与 3 个干扰项。
  2. 将每条转换为 MRAG-Bench JSONL 条目（字段 `id`,`question`(含 Choices),`answer`,`image_files`）。
  3. 把转换后数据作为本地 dataset（`local/infoseek_mrag`）供 `load_dataset` 使用，或直接把 JSONL 传入已有评估脚本。
  4. 运行 MRAG-Bench 流程（检索 + LLaVA 回答 + score.py）。

- 优点：与 MRAG-Bench 指标一致（直接可比），实现成本中等，结果可解释。
- 缺点：需要 GT（人工或 LLM+人工审核），并可能改变原任务性质（从开放到多选）。

### 方案 B（保留/研究线）：原生开放式评估

- 对 InfoSeek 保持开放问答形式，用语义评分或 LLM judge：
  - 计算答案与人工参考文本的相似度（E5/SBERT）
  - 或用高质量 LLM 给出二元判定（正确/错误）或评分（0-1）

- 优点：保持任务原生属性，评估生成能力。
- 缺点：评分主观性、难以与 MRAG-Bench 直接对比。

### 方案 C（辅助）：把 InfoSeek 图像加入检索语料库（corpus 扩容）

- 把 InfoSeek 的 `images/all`（1,005,415 张）作为 MRAG 的额外 corpus，评估检索器在更大库下的表现。
- 优点：最小改动即可评估 corpus scale 的影响。
- 缺点：需要大量存储/IO，评估成本上升。


### 推荐组合

- 开始阶段并行推进：
  - 主线：方案 A（对一小批样本完成 GT 标注，跑 MRAG-Bench 对标实验）
  - 辅助：方案 C（在远程服务器上把 InfoSeek 图像作为 corpus 扩容，评估检索伸缩性）
  - 后续（研究）：方案 B（开放式评估与语义评分）

理由：方案 A 保证结果可比、可复现；方案 C 快速验证 corpus 尺度影响；方案 B 用于深入研究生成能力与真实开放问答场景。

---

## 四、示例：把一条 InfoSeek 记录转换为 MRAG JSONL（示例脚本思路）

伪代码：

```python
# input: record = {data_id, image_id, question}
# step1: get or generate correct_answer + 3 distractors
# step2: format question with choices
mrag_item = {
  'id': record['data_id'],
  'question': f"{record['question']}\n Choices:\nA: {A}\nB: {B}\nC: {C}\nD: {D}",
  'answer': 'A',
  'image_files': [f"/mnt/infoseek_images/{record['image_id']}.jpg"]
}
# write to jsonl
```

保存后，使用 MRAG 的评估脚本跑小样本并用 `score.py` 得到 Accuracy。

---

## 五、建议的短期执行计划（可复制的 1-2 天任务）

1. 从 `data/infoseek/Entity/infoseek_test.jsonl` 抽取 200 条样本。
2. 对 200 条做人工标注（或半自动：LLM 生成 + 人工审查）以获得 GT。
3. 编写 `src/mrag/infoseek_converter.py` 完成批量转换并输出 `data/infoseek/infoseek_mrag.jsonl`。
4. 运行 MRAG-Bench 的 smoke test（`MAX_SAMPLES=20`），记录结果并对比 baseline。

---

## 参考命令速查

```bash
# 验证元数据
python3 test/test_infoseek_metadata.py

# 生成 100 条 image_id 样本
head -100 data/infoseek/Entity/infoseek_test.jsonl > /tmp/infoseek_100.jsonl

# 运行 smoke 测试（示例）
MAX_SAMPLES=20 bash test/benchmark_magiclens.sh
```

---

文档结束。若需要，我可以：
- 直接实现 `src/mrag/infoseek_converter.py`（含 LLM 调用与缓存）；
- 或帮你准备用于人工标注的 CSV/LabelStudio 导出格式；
- 或在服务器上跑一次小规模的远程评估并回传结果。