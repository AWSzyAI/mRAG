# Sync Module Package

这个目录现在就是 sync 模块本体，不再只是“参考实现”。目标是：

- 把远程 sync 相关实现全部收敛到 `sync/`
- 新项目只需要复制整个 `sync/`
- 根目录 `Makefile` 只负责 `include sync/Makefile`

## 当前模块结构

- `Makefile`: sync 主实现（`sync/sync-env/env-smoke/pull/cmd/config/role/bpe/clean`）
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

1. 复制整个 `sync/` 到新项目根目录
2. 在新项目根目录新建一个最薄的 `Makefile`
3. 写入：

```make
PROJECT_ROOT := $(CURDIR)
include sync/Makefile
```

4. 按项目需要修改：
   - `sync/.sync_ssh`
   - `sync/.exclude`
   - `sync/pull_list.txt`
   - `sync/result.txt`
   - `sync/Makefile` 中默认的 `SYNC_HOST`（如果需要）

## 当前使用方式

在项目根目录执行：

- `make sync`
- `make sync y`
- `make sync-env`
- `make sync-env y`
- `make env-smoke`
- `make pull`
- `make pull y`
- `make pull result y`
- `make cmd CMD='...'`
- `make alias`

虽然命令还是在项目根目录运行，但实际实现和配置都来自 `sync/`。

## Conda 环境同步

详细操作、离线 wheelhouse、只同步 `scenic` 源码等情况见 `../README/NNU_OFFLINE_ENV_SYNC.md`。

服务器无法联网时，不要在远端执行 `pip install .` 或 `conda env create`。推荐在本地把 `llava` 环境配好，再用 `conda-pack` 打包上传：

```bash
# 预览：只显示本地/远端路径，不创建归档
make sync-env

# 执行：打包本地 llava，上传到 sync/.sync_ssh 中的 CONDA_ENV
make sync-env y

# 如果远端目标环境已经存在，会默认停止；确认要替换时使用：
make sync-env y ENV_REPLACE=1

# 单独验证远端环境和源码依赖是否可 import
make env-smoke
```

默认会额外同步这些源码依赖到远端项目目录，因为它们通常被普通代码同步排除：

```text
github/LLaVA-NeXT github/magiclens github/scenic
```

可按需覆盖：

```bash
make sync-env y LOCAL_CONDA_ENV=llava ENV_REMOTE_PATH=/home/user/env/envs/llava
make sync-env y ENV_CODE_DEPS='github/LLaVA-NeXT github/magiclens github/scenic'
```

## 约束提醒

- `pull` 默认 dry-run，必须显式尾参 `y` 才执行。
- `make pull -y` 不可用，请使用 `make pull y`。
- `.sync_ssh` 已在 `sync/.exclude` 中排除，默认不会被同步到远端。
- 当前根目录不再保存 `.sync_ssh`、`.exclude`、`.alias`、`result.txt` 这些 sync 配置文件。
