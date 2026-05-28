# InfoSeek 数据集使用指南

> **一句话定位**: 多模态信息寻求基准 - 给定查询+图片库，测试模型的图像检索和信息推理能力

## 现象层：数据集概览

### 核心统计
- **总图片数**: 1,005,415 张（去重后）
  - Entity 来源: 1,001,028 张
  - Human 来源: 8,884 张
- **任务维度**: 实体识别 + 人工评测 + 查询
- **数据格式**: JSONL（逐行JSON）

### 关键特征
- **绑定字段**: `image_id` 是数据与图片的唯一连接点
- **统一图片库**: `infoseek/images/all/` 消除 entity/human 的二元性
- **断点续跑**: 脚本支持中断重启，自动跳过已处理图片

---

## 本质层：系统架构

### 数据流向图
```
OVEN 原始图库 (ychenNLP/oven)
    ↓
    └─→ Entity/*.jsonl (1001028 个 image_id) ─┐
        Human/*.jsonl (8884 个 image_id)    ├→ 去重 → image_id 集合
        Query/*.jsonl (无 image_id)         │
                                            ↓
                            筛图脚本 select_infoseek_images.py
                                            ↓
                            infoseek/images/all/ (1005415 张图)
```

### 目录结构职责

| 路径 | 内容 | 用途 |
|------|------|------|
| `Entity/*.jsonl` | 1001028 条记录 | 结构化实体任务数据（含 image_id） |
| `Human/*.jsonl` | 8884 条记录 | 人工评测基准数据（含 image_id） |
| `Query/*.jsonl` | 查询集 | 问题/任务定义（无 image_id） |
| `images/all/` | 1005415 张文件 | 所有需要的图片，直接按 image_id 查找 |
| `select_infoseek_images.log` | 执行日志 | 筛图过程的统计和错误记录 |

**关键设计决策**:
- ✅ **去重**: Entity 和 Human 的 image_id 去重，避免冗余存储
- ✅ **单一图库**: `images/all/` 一个目录替代多个分类目录，消除查询复杂性
- ❌ **不处理 Query**: Query/*.jsonl 中缺少 image_id，脚本未涉及

---

## 哲学层：使用范式

### 核心原则

**原则 1: 以 image_id 为真相源**
```python
# ✅ 正确做法：查询图片时使用 image_id
image_id = record['image_id']  # 从 Entity/Human 数据读取
image_path = f'/mnt/d/mRAG/infoseek/images/all/{image_id}.jpg'

# ❌ 错误做法：自己维护图片索引映射
# 违反单一真相源原则
```

**原则 2: 脚本幂等性**
```bash
# 可以安全重复执行，自动跳过已完成的图片
python3 infoseek/scripts/select_infoseek_images.py --root /mnt/d/mRAG
# 第一次: 处理所有 1005415 张
# 第二次: 0 张（都已存在，10 秒内完成）
```

**原则 3: 日志即验收**
```bash
# 最后一行必须是这个格式才算成功
tail -1 infoseek/images/select_infoseek_images.log
# all: requested=1005415, found=1005415, missing=0
```

---

## 实战流程

### 场景 1: 从零开始准备数据集

```bash
# 第 1 步：准备 OVEN 原始图库（二选一）

# 方式 A：自动下载整理（首选，自含脚本）
cd /mnt/d/mRAG
bash infoseek/scripts/download_oven_full.sh /mnt/d/mRAG

# 方式 B：手动解压预下载的分片
cd /mnt/d/mRAG/oven
mkdir -p image_downloads/oven_images
for tar in shard{01..08}.tar; do
  tar -xf "$tar" -C image_downloads/oven_images
done

# 第 2 步：筛选 InfoSeek 需要的图片
cd /mnt/d/mRAG
python3 infoseek/scripts/select_infoseek_images.py --root /mnt/d/mRAG \
  | tee infoseek/images/select_infoseek_images.log

# 第 3 步：验收
tail -20 infoseek/images/select_infoseek_images.log
# 检查是否出现：all: requested=1005415, found=1005415, missing=0
```

**耗时估算**:
- OVEN 下载: ~2-4 小时（取决于网络）
- 图片筛选: ~30-60 分钟（1M 图片量级）

### 场景 2: 检查筛图是否完整

```bash
# 快速检查：看日志末尾
tail -1 infoseek/images/select_infoseek_images.log

# 精确检查：运行覆盖率脚本
python3 infoseek/scripts/check_infoseek_image_coverage.py --root /mnt/d/mRAG

# 文件数统计
find infoseek/images/all -type f | wc -l
# 预期: 1005415
```

### 场景 3: 在代码中使用 image_id 查找图片

```python
import json
from pathlib import Path

# 读取数据
with open('infoseek/Entity/xxx.jsonl') as f:
    for line in f:
        record = json.loads(line)
        image_id = record['image_id']
        
        # 按 image_id 查找图片
        image_path = Path('/mnt/d/mRAG/infoseek/images/all') / f'{image_id}.jpg'
        
        # 使用图片
        if image_path.exists():
            img = load_image(image_path)
            # ... 处理图片
```

### 场景 4: 中断后继续筛图

```bash
# 假设脚本因网络/磁盘问题中断
# 直接重新执行同一条命令，脚本会自动：
# 1. 重新加载 OVEN 图片索引（缓存在 oven/image_downloads/oven_images_index.csv）
# 2. 检查 infoseek/images/all/ 中已完成的图片
# 3. 继续处理剩余的 image_id

python3 infoseek/scripts/select_infoseek_images.py --root /mnt/d/mRAG \
  | tee -a infoseek/images/select_infoseek_images.log
```

---

## 三种脚本的职责

### 1. `download_oven_full.sh`
**职责**: 从官方源下载并整理 OVEN 原始图库

**输入**: 仓库根目录路径  
**输出**: `/mnt/d/mRAG/oven/image_downloads/oven_images/`  
**幂等性**: ✅ 支持重复执行（已有文件跳过）

### 2. `select_infoseek_images.py`
**职责**: 筛选 Entity + Human 需要的图片，去重后统一写入

**流程**:
1. 读取 `Entity/*.jsonl` 和 `Human/*.jsonl`，收集所有 `image_id`
2. 在 `oven/image_downloads/oven_images/` 中查找每个 `image_id`
3. 优先硬链接，失败时复制到 `infoseek/images/all/`
4. 每 5000 张打印进度，最后汇总统计

**幂等性**: ✅ 自动跳过已存在图片  
**日志**: `infoseek/images/select_infoseek_images.log`

### 3. `check_infoseek_image_coverage.py`
**职责**: 验证数据完整性——检查所有 Entity/Human 记录中引用的图片是否都已到位

**输出示例**:
```
Entity coverage: 1001028/1001028 (100.0%)
Human coverage: 8884/8884 (100.0%)
Total coverage: 1005415/1005415 (100.0%)
```

---

## 常见问题

### Q: 为什么要去重？Entity 和 Human 有重复的 image_id 吗？

**A**: 有的。Entity 是自动化标注，Human 是人工评测，两个数据源对同一张图片的标注可能不同。去重后在 `images/all/` 中只保留一份物理文件，节省磁盘空间，但代码中需要根据数据来源（Entity vs Human）选择对应的标注。

### Q: Query/*.jsonl 为什么不处理？

**A**: Query 数据中没有 `image_id` 字段，所以无法关联图片。Query 可能用于定义任务的问题模板，但具体的图片关联是通过 Entity 和 Human 建立的。

### Q: 如果网络断掉，图片筛选中断了怎么办？

**A**: 脚本自动跳过已完成的图片，重新执行同一条命令即可继续。脚本在 `oven/image_downloads/oven_images_index.csv` 中缓存了 OVEN 图片索引，第二次运行会更快。

### Q: 硬链接和复制有什么区别？

**A**: 
- **硬链接**: 0 额外磁盘占用，多个文件指向同一物理块
- **复制**: 占用磁盘空间，但如果 OVEN 原始数据被删除，InfoSeek 图片不受影响

脚本优先硬链接（快），失败时退化为复制（安全）。

### Q: 1005415 张图片占多少磁盘空间？

**A**: 平均每张 ~100-200 KB（JPEG），总计约 **100-200 GB**。具体取决于图片质量。

---

## 数据格式参考

### Entity 数据格式（示例）
```json
{
  "image_id": "xxxxx",
  "entity": "某实体名",
  "question": "关于该实体的问题",
  "context": "额外上下文",
  "annotation": "标注结果"
}
```

### Human 数据格式（示例）
```json
{
  "image_id": "yyyyy",
  "query": "人工评测问题",
  "rating": 5,
  "feedback": "评测反馈"
}
```

### Query 数据格式（示例）
```json
{
  "query_id": "q001",
  "question": "查询问题",
  "query_type": "entity|open-ended|..."
}
```
*注意*: Query 中**无** `image_id`，不直接关联图片

---

## 下一步

### 立即需要的
- [ ] 从 nnu 同步 `infoseek/Entity/` 和 `infoseek/Human/` 数据文件
- [ ] 确认 OVEN 图片库已准备（方式 A 或 B）
- [ ] 执行 `select_infoseek_images.py` 筛图
- [ ] 验证 `infoseek/images/all/` 中有 1005415 张图片

### 代码集成
- [ ] 在数据加载器中，以 `image_id` 为 key 查找 `infoseek/images/all/{image_id}.jpg`
- [ ] 处理 Entity/Human 两种数据源，根据源头选择对应的标注
- [ ] 可选：实现多线程图片加载加速

---

## 参考命令速查

```bash
# 初始化
bash infoseek/scripts/download_oven_full.sh /mnt/d/mRAG

# 筛图
python3 infoseek/scripts/select_infoseek_images.py --root /mnt/d/mRAG \
  | tee infoseek/images/select_infoseek_images.log

# 验证
python3 infoseek/scripts/check_infoseek_image_coverage.py --root /mnt/d/mRAG

# 统计
find infoseek/images/all -type f | wc -l

# 看日志
tail -20 infoseek/images/select_infoseek_images.log
```

---

**文档版本**: 2026-05-11  
**GEB PROTOCOL**: 代码即文档，文档即代码。此文档与实际脚本行为应同构。  
[PROTOCOL]: 若脚本或数据结构变更，立即同步更新此文档。
