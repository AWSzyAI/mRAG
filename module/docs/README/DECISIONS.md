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

## ADR-006: 统一 pull 清单机制（预览默认，`y` 执行）

**日期**: 2026-02-14
**状态**: ✅ 已采纳

### 背景
原有 pull 逻辑按目标拆分（如仅 result），难以覆盖 AC 临时改代码回流到 M2 的场景，也不利于在拉取前评估变更规模。

### 决策
将 pull 统一为清单驱动:

- `make pull` 使用 `pull_list.txt` 进行 dry-run 预览。
- `make pull y` 才实际下载。
- `make pull result y` 使用 `result.txt` 拉取结果产物。
- `mr` 别名映射为 `make pull result y`。

### 实现
- `Makefile` 的 `pull` 目标支持:
  - list 选择: `<name>.txt`（如 `result` -> `result.txt`）
  - 执行开关: `y`（推荐）或 `APPLY=1`（兼容）
  - 输出: `rsync --itemize-changes --stats` 变更规模
- 新增 `pull_list.txt` 与 `result.txt`。

### 后果
- ✅ 拉取前可见差异规模，降低误操作风险
- ✅ AC 临时改动可按路径清单回流到 M2
- ✅ 不再依赖单独的 `pull-code` / `pull-results` 工作流
- ⚠️ 维护者需保证 pull 清单文件及时更新

---

## ADR-007: ROLE 元数据纳入 `make config`

**日期**: 2026-02-14
**状态**: ✅ 已采纳

### 背景
AI 在不同节点（M2 主开发端 vs AC/NNU 服务器端）会看到不同上下文，缺少显式角色信息易导致误判。

### 决策
把 ROLE 文档初始化并入 `make config`：

- 本地生成 `.agent/ROLE.md`
- 远端生成 `$(REMOTE_DIR)/.agent/ROLE.md`
- 回流远端角色到本地 `.agent/ROLE.<SYNC_HOST>.md`

### 后果
- ✅ AI 对话可以快速识别当前节点角色
- ✅ 新增服务器时可自动生成对应角色文档
- ⚠️ 需要 `.sync_ssh` 提供完整 `SYNC_HOST/REMOTE_DIR/CONDA_ENV`

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
