# InfoSeek Sync Status

更新日期：2026-05-20

## 当前结论

本地 InfoSeek 元数据已经存在，并且刚执行过一次真实 pull：

```bash
make pull infoseek y
```

执行结果显示远端没有新的文件需要下载：

```text
Changed entries: 0
Files to pull: 0
Rsync regular files transferred: 0
```

这意味着：`sync/infoseek.txt` 清单里的文件在当前远端和本地之间没有检测到差异。此次 pull 没有写入新的 `data/infoseek` 文件。

## 使用的同步配置

当前根目录 `Makefile` include 的是：

```make
include sync/Makefile
```

`sync/Makefile` 默认：

```make
SYNC_HOST ?= nnu
```

对应 `sync/.sync_ssh` 中的远端配置：

```text
Host nnu
  REMOTE_DIR ?= /home/user/code/mRAG
  CONDA_ENV ?= /home/user/env/envs/llava
```

## 本地数据状态

本地元数据文件存在，行数如下：

```text
347980  data/infoseek/Entity/infoseek_test.jsonl
934048  data/infoseek/Entity/infoseek_train.jsonl
 73620  data/infoseek/Entity/infoseek_val.jsonl
  8931  data/infoseek/Human/infoseek_human.jsonl
934048  data/infoseek/Query/infoseek_train_withkb.jsonl
 73620  data/infoseek/Query/infoseek_val_withkb.jsonl
2372247 total
```

当前 `data/infoseek` 总大小约 `410M`。

## 图片状态

本地 `data/infoseek/images` 当前只有样例图片：

```text
66 files
8.8M data/infoseek/images
```

这不是完整 InfoSeek 图片库。完整图片库如果需要用于大规模 benchmark，需要另外拉取、挂载或在远端直接运行。

## 仍缺失或需确认的内容

`sync/infoseek.txt` 中列出了以下索引/日志目标：

```text
data/infoseek/images/select_infoseek_images.log
data/infoseek/images/oven_images_index.csv
```

但当前本地没有这两个目标文件。现在只发现一个备份日志：

```text
data/infoseek/.backup/select_infoseek_images.log
```

由于 `make pull infoseek y` 显示 `Files to pull: 0`，说明当前远端 `nnu:/home/user/code/mRAG` 上按该清单也没有可同步的新版本，或远端状态与本地状态在 rsync 判定上没有差异。

## 2026-05-20 10k 远端运行结果

已在远端 `nnu:/home/user/code/mRAG` 运行：

```bash
SAMPLE_SIZE=10000 SPLIT=entity_test RANDOM_SEED=42 \
OUTPUT_DIR=log/E11_4_infoseek_10k \
bash test/E11_4_infoseek_10k.sh
```

为了避免当前工作区的 `module -> sync` 迁移状态触发全量同步删除，远端运行前只窄范围上传了 3 个脚本：

```text
scripts/prepare_infoseek_10k_samples.py
test/E11_4_infoseek_10k.sh
test/benchmark_e11_4_infoseek.py
```

远端生成的结果已拉回本地：

```text
log/E11_4_infoseek_10k/REPORT.txt
log/E11_4_infoseek_10k/sampling/sample_indices.json
log/E11_4_infoseek_10k/sampling/sample_metadata.json
log/E11_4_infoseek_10k/sampling/samples.jsonl
log/E11_4_infoseek_10k/sampling/summary.json
log/E11_4_infoseek_10k/benchmark/e11_4_infoseek_results.jsonl
log/E11_4_infoseek_10k/benchmark/e11_4_infoseek_summary.json
```

结果规模验证：

```text
10000 log/E11_4_infoseek_10k/sampling/samples.jsonl
10000 log/E11_4_infoseek_10k/benchmark/e11_4_infoseek_results.jsonl
```

采样摘要：

```json
{
  "experiment": "E11_4_InfoSeek_10K",
  "dataset": "entity_test",
  "total_in_split": 347980,
  "sampled_count": 10000,
  "random_seed": 42
}
```

benchmark 摘要：

```json
{
  "experiment": "E11_4_InfoSeek_10K",
  "dataset": "entity_test",
  "total_samples": 10000,
  "completed": 10000,
  "failed": 0,
  "exact_match_rate": 0.0,
  "fuzzy_match_rate": 0.0
}
```

重要说明：当前 `test/benchmark_e11_4_infoseek.py` 是流程框架/占位实现。它会生成 4 维 query 字段，但检索阶段没有真正调用 MagicLens，答案阶段没有真正调用 LLaVA，因此结果中 `retrieval_results` 为空、`predicted_answer` 为空，0% accuracy 不能作为真实模型效果解读。

本次新增了结果 pull 清单：

```text
sync/infoseek_10k_results.txt
```

按理可以使用：

```bash
make pull infoseek_10k_results y
```

但当前 `make pull` 对 `log/` 路径有保护逻辑，目录级或部分 log 清单没有把文件落地。此次最终使用了精确 `rsync` 拉取 7 个产物文件。

## 下一步

如果目标是完整 E11_4 InfoSeek benchmark，需要把 `test/benchmark_e11_4_infoseek.py` 从占位实现接入真实链路：

1. 在远端直接运行 benchmark，使用远端完整图片路径。
2. 接入 MagicLens 检索，写入 `retrieval_results`。
3. 接入 LLaVA/Gemma4 答案生成，写入 `predicted_answer`。
4. 准备参考答案或评分逻辑，否则开放式 QA 的 EM/fuzzy match 仍无法反映真实效果。

注意：完整图片目录体积很大，不建议在没有确认磁盘空间和实验策略前直接全量 pull。

## 2026-05-20 真调用启动尝试

已将 `test/benchmark_e11_4_infoseek.py` 从占位版改为真实调用版，新增能力：

```text
--retriever magiclens|none
--dim-generator-type gemma4_local|raw_question|heuristic
--final-answerer gemma4|llava|none
--resume-from-existing
```

当前服务器验证结果：

1. `python test/benchmark_e11_4_infoseek.py --help` 可正常运行。
2. `data/infoseek/images/all/oven_05623354.jpg` 存在，完整图片目录路径可读。
3. `github/magiclens` 与 `github/LLaVA-NeXT` 是 gitlink/submodule 指针，但当前本地和远端都没有完整源码。
4. 已从 GitHub 下载并同步 MagicLens 源码到远端 `github/magiclens/`，但在当前远端环境中 `import inference` 仍会导致进程断开。
5. LLaVA-NeXT 远端目录也缺源码，只剩 `__pycache__`，因此 `from llava.constants import ...` 无法正常导入。
6. Gemma4 在当时默认 `py310` 环境中被 PyTorch 版本挡住：

```text
torch=2.1.2+cu121
Gemma4 loader requires torch>=2.4 for the installed transformers stack.
```

强制加 `--gemma4-allow-torch-below-2-4` 后，进程在 TensorFlow/transformers 初始化后断开，未进入推理。

因此目前“真实调用”还没有成功跑出第一条结果。需要先修服务器运行环境：

1. 恢复/签出 `github/LLaVA-NeXT` 源码。
2. 让 MagicLens 在某个 conda 环境中可以稳定 `import inference`。
3. 为 Gemma4 准备 torch>=2.4 的环境，或改用一个当前环境能加载的 VLM answerer。
