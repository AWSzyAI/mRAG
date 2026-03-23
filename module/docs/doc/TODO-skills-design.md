# TODO: repo-server-sync Skill 设计

## 1. 背景与目标

目标是把当前 mRAG 的 `repo <-> server` 工作流抽象成一个可复用 Skill，用于任意项目快速落地以下能力：

- 主开发端（如 M2）为代码 source of truth。
- 多服务器（如 AC/NNU）作为运行端，可按需临时改代码并回流。
- `make pull` 默认预览，`make pull y` 才执行。
- 通过 pull 清单控制拉取范围（代码/结果分离）。
- 通过 ROLE 文档让 AI 快速识别当前节点身份。

## 2. 非目标

- 不在 Skill 内硬编码 AC/NNU/M2 这类具体主机名。
- 不绑定特定项目结构（例如必须有 MRAG-Bench）。
- 不替代 git 工作流（Skill 仅解决同步与环境识别）。

## 3. 核心设计原则

- 配置驱动：项目差异通过配置文件表达，而不是改 Skill 代码。
- 交互式初始化：首次使用通过问答收集最小必要信息。
- 幂等更新：重复执行只补齐缺失项，不粗暴覆盖已有手工配置。
- 安全默认：pull 默认 dry-run；执行必须显式 `y`。

## 4. Skill 启动流程（对话式向导）

### 4.1 首次启动

Skill 首次在某个 workspace 运行时，按顺序询问：

1. 当前目标 repo 路径（默认当前目录）
2. 主开发端标识（例如 `main-dev` / `M2`）
3. 服务器列表（可多选/多次输入）
4. 每台服务器对应：
   - ssh host（对应 `~/.ssh/config` 的 Host）
   - 远端项目目录 `REMOTE_DIR`
   - 远端 conda 环境名（可选）
5. 默认拉取清单文件名（默认 `pull_list.txt`）
6. 结果拉取清单文件名（默认 `result.txt`）
7. 是否初始化 ROLE 文档与 `make config` 挂钩（默认是）

### 4.2 二次启动

- 若已存在项目配置，Skill 不重复完整提问。
- 仅在缺字段时追问。
- 支持命令式增量操作：
  - 新增服务器
  - 修改某服务器 `REMOTE_DIR`
  - 重设默认 pull 清单

## 5. 项目内配置契约（建议）

建议在项目内生成：`doc/sync-contract.yaml`

建议字段（草案）：

- `version`
- `main_dev`
- `default_pull_list`
- `result_pull_list`
- `hosts[]`:
  - `name`
  - `ssh_host`
  - `remote_dir`
  - `conda_env`
  - `role_kind`（一般为 `server-runtime`）

说明：Skill 后续所有生成动作均以该文件为真源。

## 6. Skill 生成/维护的目标文件（草案）

- `Makefile`（增量注入同步目标，不破坏业务 target）
- `.alias`（如 `ms/mc/mr`，可开关）
- `pull_list.txt`
- `result.txt`
- `.exclude`（仅追加必要排除项）
- `.agent/ROLE.md`（本机）
- `README` 片段（可选）

## 7. 命令语义标准（Skill 统一约定）

- `make sync`：本地 -> 远端同步
- `make pull`：预览（dry-run）
- `make pull y`：执行默认清单拉取
- `make pull <name>`：预览 `<name>.txt`
- `make pull <name> y`：执行 `<name>.txt`
- `make pull -y`：明确标注为不支持（GNU make 参数冲突）

## 8. ROLE 规范（AI 环境识别）

- 本机角色：`.agent/ROLE.md`
- 远端角色：`<REMOTE_DIR>/.agent/ROLE.md`
- 回流角色：`.agent/ROLE.<HOST>.md`
- `make config` 完成后应自动完成角色初始化/回流

## 9. 验收标准

- 新项目首次运行 Skill 后，可直接执行：
  - `make sync`
  - `make pull`
  - `make pull y`
- 至少支持 2 台服务器配置并可切换。
- 文档中明确 `y` 语义与 `-y` 限制。
- 不破坏原有项目业务构建流程。

## 10. 迁移计划（现有项目接入）

1. 读取并备份现有 `Makefile/.alias/.exclude`
2. 生成 `doc/sync-contract.yaml`
3. 以最小 diff 注入目标
4. 执行 dry-run 验证
5. 人工确认后再 apply

## 11. 未决问题

- 是否要求 Skill 自动写入 `~/.ssh/config`，还是仅校验并提示手工处理？
- 多项目共用同一服务器时，`REMOTE_DIR` 命名规范是否统一？
- 是否需要 `pull` 审计日志（每次变更文件清单落盘）作为默认行为？
- `mr` 这类 alias 是否默认开启，还是按项目开关？
