# InfoSeek 集成 - 快速参考

## 🎯 三行代码开始

```bash
# 1. 转换 InfoSeek 问题到 MRAG 格式
python3 -c "
from src.mrag.infoseek_converter import InfoSeekConverter
converter = InfoSeekConverter()
converter.batch_convert('data/infoseek/Entity/infoseek_test.jsonl', 'results/converted.jsonl', max_samples=100)
"

# 2. 本地模式评估 (秒级完成)
python test/benchmark_infoseek.py --mode local --input-jsonl results/converted.jsonl --max-samples 100

# 3. 混合模式评估 (带真实图像)
python test/benchmark_infoseek.py --mode hybrid --input-jsonl results/converted.jsonl --image-dir data/infoseek/images --max-samples 66
```

## 📊 核心数据

| 指标 | 值 |
|------|-----|
| 元数据总记录 | 2.3M |
| 本地样本图像 | 66 |
| 图像覆盖率 | 70% |
| 本地图像体积 | 8.8M |
| 转换器缓存 | SQLite |
| 支持 LLM | OpenAI / Local |

## 🔧 核心组件

| 文件 | 行数 | 功能 |
|------|------|------|
| `src/mrag/infoseek_converter.py` | 500+ | 问题转换 (LLM + 缓存) |
| `test/benchmark_infoseek.py` | 400+ | 三层评估框架 |
| `test/test_infoseek_converter.py` | 200+ | 转换器单元测试 |

## 📈 工作流

```
InfoSeek 问题
  ↓
转换器 (LLM multi-choice)
  ↓
MRAG 格式 {options: A/B/C/D, correct: "A"}
  ↓
评估器 (3 种模式)
  ├─ Local (纯逻辑)
  ├─ Hybrid (66 图)
  └─ Remote (完整)
  ↓
准确率 + 检索指标
```

## ✅ 测试状态

- [x] 元数据加载: 2.3M 记录验证通过
- [x] 转换器: 单元测试通过
- [x] 本地评估: 10 个样本 100% 准确
- [x] 混合评估: 70% 图片覆盖率验证
- [ ] MagicLens 集成 (规划)
- [ ] LLaVA 集成 (规划)
- [ ] 远程完整评估 (规划)

## 🚀 快速命令

```bash
# 查看转换器帮助
python src/mrag/infoseek_converter.py -h

# 转换首 1000 个样本 (需要 OpenAI API key)
export OPENAI_API_KEY="sk-..."
python src/mrag/infoseek_converter.py data/infoseek/Entity/infoseek_test.jsonl \
    -o results/converted.jsonl \
    -n 1000 \
    -m gpt-3.5-turbo

# 查看本地评估帮助
python test/benchmark_infoseek.py --help

# 混合模式评估，66 个样本
python test/benchmark_infoseek.py \
    --mode hybrid \
    --input-jsonl results/converted.jsonl \
    --image-dir data/infoseek/images \
    --max-samples 66 \
    --output-dir results/infoseek

# 查看评估结果
python -m json.tool results/infoseek/results_hybrid.json
```

## 🔐 环境变量

```bash
# OpenAI API 配置 (转换器需要)
export OPENAI_API_KEY="sk-..."

# 可选: 缓存数据库位置
export INFOSEEK_CACHE_DB="/tmp/infoseek.db"
```

## 📦 文件位置速查

```
✅ 元数据: /mnt/d/mRAG/data/infoseek/{Entity,Human,Query}/*.jsonl
✅ 样本图像: /mnt/d/mRAG/data/infoseek/images/ (66个)
✅ 转换器: /mnt/d/mRAG/src/mrag/infoseek_converter.py
✅ 评估: /mnt/d/mRAG/test/benchmark_infoseek.py
📝 文档: /mnt/d/mRAG/README/INFOSEEK_*.md
```

## 🆘 常见问题

**Q: 转换很慢？**  
A: 检查缓存是否启用，第一次需要调用 LLM，后续从缓存读取很快

**Q: 内存溢出？**  
A: 减小 `max_samples` 或增加 `batch_size` 参数

**Q: 没有 OpenAI 密钥？**  
A: 使用 `--model local` 使用本地模型替代

**Q: 图片找不到？**  
A: 本地只有 66 个样本图像，前 100 个样本中覆盖率 70%

## 📚 详细文档

- `INFOSEEK_DATASET_GUIDE.md` - 数据格式详解
- `INFOSEEK_INTEGRATION.md` - 三种集成方案
- `INFOSEEK_INTEGRATION_COMPLETE.md` - 完整指南
- `test/benchmark_infoseek.py` - 源代码注释
- `src/mrag/infoseek_converter.py` - 源代码注释

---

**提示**: 建议先运行本地模式快速验证，再尝试混合/远程模式
