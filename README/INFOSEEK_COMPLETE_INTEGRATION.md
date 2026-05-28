# InfoSeek 集成到 MRAG-Bench 基础设施 - 完整方案

## 现状总结

### ✅ 已完成
1. **元数据同步本地** (1.4GB)
   - Entity: 1,001,028 条记录（test/train/val）
   - Human: 8,931 条记录（人工评测基准）
   - Query: 1,007,668 条记录（entity 背景知识）
   
2. **数据加载框架** 
   - `src/mrag/infoseek_loader.py` - JSONL 迭代器 + 图像加载接口
   - `test/test_infoseek_metadata.py` - 元数据验证脚本

3. **集成方案设计**
   - 方案 A（推荐）：转换为多选题，复用 MRAG-Bench 评估框架
   - 方案 B（可选）：开放式评估，语义评分
   - 方案 C（扩展）：作为图像语料库扩展

### ⚠️ 关键约束
- **图像文件本地缺失**（57GB 太大未 pull）
  - 远程位置: `/home/user/code/mRAG/data/infoseek/images/all/` (~1M 张)
  - 本地方案: 三种部署模式应对

---

## 三种部署模式

### 模式 1️⃣：本地开发（推荐 - 快速迭代）

**特点**: 仅测试代码逻辑，不涉及实际图像加载

**工作流**:
```python
# 1. 加载元数据
from src.mrag.infoseek_loader import InfoSeekDataset
ds = InfoSeekDataset("/mnt/d/mRAG/data/infoseek")
for record in ds.iter_split("entity_test", load_image=False):
    # record.question 可用
    # record.query_image_path 为 None（本地没图片）
    pass

# 2. LLM 生成多选项（不需要图片）
# 调用 OpenAI API / 本地 LLM 生成干扰项

# 3. 转换为 MRAG 格式
# 构造 {"id": ..., "question": ..., "choices": {...}, "answer": ...}

# 4. Mock 答案生成（模拟 LLaVA）
# 返回随机答案或固定答案进行流程验证
```

**优点**: 快速反馈，无 GPU 需求  
**缺点**: 无法验证实际检索和答案质量  
**适用**: 代码开发、格式调试、逻辑验证

---

### 模式 2️⃣：混合模式（生产前验证）

**特点**: 本地代码 + 通过 NFS/SSH 访问远程图片

**配置选项**:

#### 选项 A：NFS 挂载（推荐 - 生产方案）
```bash
# 在本地主机上
mkdir -p /mnt/nfs_infoseek
sudo mount -t nfs nnu:/home/user/code/mRAG/data/infoseek/images /mnt/nfs_infoseek

# 在代码中
export INFOSEEK_IMAGE_DIR=/mnt/nfs_infoseek/all
# 加载器会自动查找
```

#### 选项 B：SSHFS 挂载
```bash
# 更简单但性能差
sshfs nnu:/home/user/code/mRAG/data/infoseek/images/all /mnt/sshfs_infoseek

export INFOSEEK_IMAGE_DIR=/mnt/sshfs_infoseek
```

**工作流**:
```bash
# 验证挂载
ls /mnt/nfs_infoseek/all | wc -l  # 应该看到百万级图片

# 运行本地 benchmark（小规模）
python3 test/benchmark_infoseek.py \
  --dataset-name infoseek \
  --split entity_test \
  --max-samples 100 \
  --corpus-dir /mnt/nfs_infoseek/all
```

**优点**: 接近生产环境，完整验证  
**缺点**: 需要网络访问，速度依赖网络  
**适用**: 集成测试、小规模验证

---

### 模式 3️⃣：远程评估（推荐 - 完整评估）

**特点**: 在 nnu 服务器上运行，本地仅处理结果

**工作流**:
```bash
# 1. 本地编写 benchmark_infoseek.py
# 2. 上传到服务器
make sync  # 同步本地代码到 nnu

# 3. 在 nnu 上运行
make cmd 'cd /home/user/code/mRAG && \
  bash test/benchmark_infoseek.sh --max-samples 1000'

# 4. 结果拉回本地
# 编辑 sync/result.txt，添加结果文件路径
make pull result y

# 5. 本地分析
python3 github/MRAG-Bench/eval/score.py \
  -i results/infoseek_results.jsonl
```

**优点**: 完整准确，充分利用服务器资源  
**缺点**: 需要网络上传/下载  
**适用**: 完整实验、论文提交、最终评估

---

## 实现路线图

### Phase 1：基础设施（1-2 天）✅ 已完成

- [x] `src/mrag/infoseek_loader.py` - 数据加载器
- [x] 元数据本地同步
- [x] 加载器测试脚本

### Phase 2：多选题转换（3-5 天）

- [ ] `src/mrag/infoseek_converter.py`
  - 调用 LLM API 生成多选题选项
  - 质量检查（选项差异度、长度等）
  - 缓存机制（避免重复调用）

- [ ] `src/mrag/infoseek_formatter.py`
  - InfoSeek record → MRAG-Bench 格式
  - 字段映射、验证、补全

### Phase 3：本地评估脚本（2-3 天）

- [ ] `test/benchmark_infoseek.py`
  - 复用 `benchmark_corpus_rag.py` 逻辑
  - 支持三种模式（本地模拟、混合、远程）
  - 进度条、日志、错误处理

- [ ] `test/benchmark_infoseek.sh`
  - 一键脚本包装
  - 参数化配置
  - 结果自动评分

### Phase 4：远程配置（1 天）

- [ ] 服务器上 benchmark_infoseek.py 的 `main.py` 集成
- [ ] 结果回传配置 (`sync/result.txt`)
- [ ] Makefile 快捷命令

### Phase 5：对标分析（可选，后期）

- [ ] 与 MRAG-Bench 指标对比
- [ ] InfoSeek 难度分析
- [ ] 数据集特性研究

---

## 代码架构（最终形态）

```
src/mrag/
├── infoseek_loader.py          ✅ 加载器（已完成）
│   └── InfoSeekDataset 类       → 迭代 JSONL + 延迟加载图像
│
├── infoseek_converter.py        📝 多选题转换（待实现）
│   ├── question_to_mqa()        → 调用 LLM 生成选项
│   ├── validate_options()       → 质量检查
│   └── cache manager            → 避免重复调用
│
└── infoseek_formatter.py        📝 格式适配（待实现）
    └── to_mrag_format()         → 转换为 MRAG-Bench 字段

test/
├── test_infoseek_metadata.py    ✅ 元数据验证（已完成）
├── benchmark_infoseek.py        📝 评估脚本（待实现）
└── benchmark_infoseek.sh        📝 一键脚本（待实现）

github/MRAG-Bench/eval/
└── score.py                     ✅ 直接复用（无需修改）
```

---

## 快速开始命令

### 验证元数据完整性
```bash
cd /mnt/d/mRAG
python3 test/test_infoseek_metadata.py
```

**预期输出**: ✅ 2.3M 条元数据记录完整，✅ 所有字段有效

### 本地模式 - 快速测试（开发中）
```bash
# 仅验证数据加载和格式转换
python3 test/benchmark_infoseek.py \
  --mode=local \
  --max-samples=10 \
  --skip-image-loading
```

### 混合模式 - NFS 挂载后
```bash
# 假设已挂载 /mnt/nfs_infoseek
python3 test/benchmark_infoseek.py \
  --mode=hybrid \
  --image-dir=/mnt/nfs_infoseek/all \
  --max-samples=100
```

### 远程模式 - 完整评估
```bash
# 在本地编辑后同步
make sync y

# 在服务器上运行
make cmd 'cd /home/user/code/mRAG && \
  python3 test/benchmark_infoseek.py --max-samples=0'

# 拉取结果
make pull result y

# 本地评分
python3 github/MRAG-Bench/eval/score.py \
  -i results/infoseek_results.jsonl
```

---

## 关键设计决策

### ✅ 为什么转换为多选题？（方案 A）

1. **复用现有框架** - 0 修改 `score.py`，直接评分
2. **可对标对比** - 与 MRAG-Bench 用统一标准
3. **LLM 成本低** - 仅在预处理时调用一次
4. **结果清晰** - 精确的 Accuracy 指标

### ⚠️ 为什么本地缺图片？

1. **57GB 太大** - 不适合本地存储
2. **按需 pull** - 支持灵活的运行模式
3. **生产实践** - 企业通常这样配置（数据中心 + 本地计算）

### 🔧 为什么设计三种模式？

1. **模式 1** - 快速开发（不需网络）
2. **模式 2** - 集成测试（NFS 挂载）
3. **模式 3** - 完整评估（充分利用服务器）

---

## 文档索引

| 内容 | 位置 |
|------|------|
| **InfoSeek 使用指南** | [README/INFOSEEK_DATASET_GUIDE.md](README/INFOSEEK_DATASET_GUIDE.md) |
| **本集成方案** | [README/INFOSEEK_INTEGRATION.md](README/INFOSEEK_INTEGRATION.md) |
| **Pull 配置** | [sync/infoseek.txt](sync/infoseek.txt) |
| **数据加载器** | [src/mrag/infoseek_loader.py](src/mrag/infoseek_loader.py) |
| **验证脚本** | [test/test_infoseek_metadata.py](test/test_infoseek_metadata.py) |

---

## 下一步行动

### 立即（本次）
- ✅ 元数据本地化
- ✅ 数据加载框架
- ✅ 架构方案设计

### 短期（1-2 周）
- [ ] 实现 `infoseek_converter.py`（LLM 多选题生成）
- [ ] 编写 `benchmark_infoseek.py`（复用 MRAG 逻辑）
- [ ] 本地小规模验证

### 中期（2-4 周）
- [ ] 远程完整评估
- [ ] 与 MRAG-Bench 对标
- [ ] 论文分析 + 优化

---

## 问题排查

### Q: 本地能跑完整评估吗？
**A**: 取决于是否能访问图片。三种方案：
1. 模式 1 - 跳过图片，验证代码逻辑
2. 模式 2 - NFS 挂载远程图片
3. 模式 3 - 完全在服务器上运行

### Q: 如何降低 LLM 调用成本？
**A**: 
- 使用缓存（已在 `infoseek_converter.py` 计划中）
- 批量调用（减少 API overhead）
- 本地 LLM（如 Mistral 7B on GPU）

### Q: 多选题的干扰项如何生成？
**A**: 多种策略（按优先级）：
1. **知识库检索** - WikiQA 中的相关答案
2. **LLM 生成** - "为这个问题生成 3 个合理但错误的选项"
3. **随机采样** - 从数据中随机选其他答案

### Q: 能在 MRAG-Bench 的 corpus 中混入 InfoSeek 图片吗？
**A**: 可以，这是"方案 C"。但需要：
1. 确保图片格式兼容（JPEG）
2. 性能测试（1.5M 图片会慢很多）
3. 结果解释清楚是"数据集扩展"不是"新方法"

---

**最后更新**: 2026-05-11  
**状态**: 基础设施完成，待实现评估脚本  
**预计周期**: 2-3 周完整集成

