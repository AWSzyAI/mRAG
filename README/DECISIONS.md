# 技术决策记录 (ADR)

## ADR-001: 使用 HuggingFace 国内镜像

**日期**: 2026-02-12
**状态**: ✅ 已采纳

### 背景
NNU 服务器访问 `huggingface.co` 经常超时或被拒绝。

### 决策
使用 `hf-mirror.com` 作为默认 HuggingFace endpoint。

### 实现
```python
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
```

### 后果
- ✅ 下载速度显著提升
- ✅ 稳定性改善
- ⚠️ 需要在所有脚本中设置

---

## ADR-002: 使用 SDPA 替代 Flash Attention

**日期**: 2026-02-12
**状态**: ✅ 已采纳

### 背景
Flash Attention 编译困难，依赖复杂（需要 CUDA、ninja 等）。

### 决策
使用 PyTorch 原生的 SDPA (Scaled Dot-Product Attention)。

### 实现
```python
# github/MRAG-Bench/eval/models/llava_one_vision.py
attn_implementation="sdpa"
```

### 后果
- ✅ 环境配置简化
- ✅ 兼容性提升
- ⚠️ 性能可能略低于 Flash Attention（待测试）

---

## ADR-003: 本地开发 + 远程运行的混合模式

**日期**: 2026-02-12
**状态**: ✅ 已采纳

### 背景
NNU 服务器不支持 VSCode Remote-SSH 的在线 AI 工具（Claude Code、CodeX）。

### 决策
采用本地编辑代码 + rsync 同步 + 远程 GPU 运行的工作流。

### 实现
- Makefile 提供 `sync` 和 `cmd` 命令
- Shell 别名 `ms` (make sync) 和 `mc` (make cmd)

### 后果
- ✅ 可以使用本地 AI 辅助工具
- ✅ 充分利用服务器 GPU
- ⚠️ 需要手动同步代码
- ⚠️ 调试略不方便

---

## ADR-004: rsync 保护远程大文件

**日期**: 2026-02-12
**状态**: ✅ 已采纳

### 背景
`github/magiclens/data/` 包含大量数据，不应被本地同步覆盖。

### 决策
使用 `.exclude` 文件排除大数据目录。

### 实现
```
# .exclude
github/magiclens/data
github/magiclens/data/***
```

### 后果
- ✅ 保护远程数据不被删除
- ✅ 减少同步时间
- ⚠️ 新增数据路径需要手动添加

---

## ADR-005: Conda 环境命名为 llava

**日期**: 2026-02-12
**状态**: ✅ 已采纳

### 背景
需要一个稳定的 Python 3.10 + PyTorch 2.1.2 环境。

### 决策
创建名为 `llava` 的 conda 环境。

### 实现
```bash
conda create -n llava python=3.10 -y
conda activate llava
```

### 后果
- ✅ 环境隔离
- ✅ 依赖版本锁定
- ⚠️ 所有远程命令需要 `conda activate llava`

---

## 待决策事项

### TBD-001: 模型文件持久化位置

**问题**: 服务器重启后 `/home/user/.cache/` 是否会被清除？

**选项**:
1. 保持默认路径 `.cache/huggingface/`
2. 迁移到 `/home/user/env/` 或项目 `./models`

**等待**: 观察下次服务器重启

---

### TBD-002: 是否集成 MagicLens 模型

**问题**: MagicLens 是否能提升 mRAG 性能？

**待测试**: 运行 MagicLens 推理并对比结果

---

### TBD-003: 优化 Incomplete 场景

**问题**: Incomplete 场景准确率仅 30.39%

**待分析**:
- 数据集特点
- 模型局限性
- Prompt 工程优化
