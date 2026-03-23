# Sync Module Package

这个目录用于把 mRAG 当前的“本地开发 + 远端运行”同步机制整体迁移到新项目。

## 目录说明

- `docs/`: 现有文档与决策记录（完整参考）
- `reference/`: 可直接复用的同步机制实现文件
- `templates/`: 新项目初始化时可改名/改内容的模板

## reference 文件用途

- `reference/Makefile`: 同步机制主实现（`sync/pull/cmd/config/role`）
- `reference/.exclude`: rsync 排除规则
- `reference/.alias`: 命令别名配置（`ms/mc/mr`）
- `reference/.sync_ssh`: 多主机同步配置（按 Host 读 `REMOTE_DIR/CONDA_ENV`）
- `reference/result.txt`: 结果回拉清单示例
- `reference/scripts/build_rsync_filter.awk`: `.exclude` -> rsync filter 转换器
- `reference/scripts/read_sync_value.awk`: 从 `.sync_ssh` 读取 host 配置

## templates 文件用途

- `templates/pull_list.txt`: 默认代码/文件回拉清单模板

## 在新项目中落地

1. 把 `reference/Makefile` 合并或替换到新项目根目录 `Makefile`。
2. 把 `reference/scripts/*.awk` 复制到新项目 `scripts/`。
3. 把 `reference/.exclude`、`reference/.alias`、`reference/.sync_ssh`、`templates/pull_list.txt`、`reference/result.txt` 放到新项目根目录。
4. 在新项目中根据实际环境修改：
   - `SYNC_HOST`
   - `.sync_ssh` 中对应 Host 的 `REMOTE_DIR` 与 `CONDA_ENV`
   - `.exclude` 的排除路径
   - `pull_list.txt` / `result.txt` 的拉取路径
5. 执行：
   - `make sync`（预览）
   - `make sync y`（应用）
   - `make pull`（预览）
   - `make pull y`（应用）

## 约束提醒

- `pull` 默认 dry-run，必须显式 `y` 才执行。
- `make pull -y` 不可用，请使用尾参 `y`。
- `.sync_ssh` 已在 `.exclude` 中排除，默认不会被同步到远端。
