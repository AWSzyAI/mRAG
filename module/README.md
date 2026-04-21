# Sync Module Package

这个目录现在就是 sync 模块本体，不再只是“参考实现”。目标是：

- 把远程 sync 相关实现全部收敛到 `module/`
- 新项目只需要复制整个 `module/`
- 根目录 `Makefile` 只负责 `include module/Makefile`

## 当前模块结构

- `Makefile`: sync 主实现（`sync/pull/cmd/config/role/bpe/clean`）
- `.sync_ssh`: 多主机同步配置
- `.exclude`: rsync 排除规则
- `.alias`: `mc/ms/mr` 别名定义
- `pull_list.txt`: 默认 pull 清单
- `result.txt`: 结果产物 pull 清单
- `scripts/`: awk 辅助脚本
- `templates/role_prompt.md`: ROLE 文档模板
- `reference/`: 旧的参考拷贝，保留作历史备份
- `docs/`: 旧文档镜像与设计材料

## 在新项目中落地

推荐的新项目接入方式：

1. 复制整个 `module/` 到新项目根目录
2. 在新项目根目录新建一个最薄的 `Makefile`
3. 写入：

```make
PROJECT_ROOT := $(CURDIR)
include module/Makefile
```

4. 按项目需要修改：
   - `module/.sync_ssh`
   - `module/.exclude`
   - `module/pull_list.txt`
   - `module/result.txt`
   - `module/Makefile` 中默认的 `SYNC_HOST`（如果需要）

## 当前使用方式

在项目根目录执行：

- `make sync`
- `make sync y`
- `make pull`
- `make pull y`
- `make pull result y`
- `make cmd CMD='...'`
- `make alias`

虽然命令还是在项目根目录运行，但实际实现和配置都来自 `module/`。

## 约束提醒

- `pull` 默认 dry-run，必须显式尾参 `y` 才执行。
- `make pull -y` 不可用，请使用 `make pull y`。
- `.sync_ssh` 已在 `module/.exclude` 中排除，默认不会被同步到远端。
- 当前根目录不再保存 `.sync_ssh`、`.exclude`、`.alias`、`result.txt` 这些 sync 配置文件。
