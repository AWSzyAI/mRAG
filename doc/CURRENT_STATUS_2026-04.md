# mRAG 项目当前状态同步（2026-04）

本文档用于同步当前仓库中 `代码`、`paper`、`doc` 与 `服务器同步脚本` 的真实状态，作为这一阶段最值得优先阅读的入口文档。

## 一句话结论

当前项目的主线已经从“单纯复现 MRAG-Bench + MagicLens 接入”转向：

`CLIP 粗召回 + MagicLens 关系检索/重排 + 多维度 query decomposition + 论文整合`

其中，代码层面的核心正在逐步沉淀到 `src/mrag/`，实验入口仍主要保留在 `test/`，论文主叙事已经更新到 `paper/content.tex`，而服务器工作流现在以 `根目录 Makefile import module/Makefile` 的模块化机制为准。

## 1. 当前代码状态

### 1.1 主代码入口已经开始模块化

当前最值得当作“核心实现”阅读的目录是：

- `src/mrag/`

其中职责大致如下：

- `src/mrag/clip_retriever.py`: CLIP 编码、query/corpus 检索
- `src/mrag/magiclens.py`: MagicLens 编码、query 检索、rerank 逻辑
- `src/mrag/indexing.py`: corpus embedding 缓存与索引构建
- `src/mrag/query_planner.py`: 多维度检索指令生成
- `src/mrag/fusion.py`: 多维度结果融合
- `src/mrag/llava.py`: LLaVA 加载与回答接口
- `src/mrag/mrag_bench.py`: MRAG-Bench 数据读取包装
- `src/mrag/runtime.py` / `src/mrag/text.py`: 日志、文本解析等公共工具

### 1.2 实验入口仍在 `test/`

虽然核心逻辑正在迁移到 `src/mrag/`，但当前实际跑实验时，最常用的入口仍然是：

- `test/benchmark_corpus_rag.py`
- `test/benchmark_magiclens.py`
- `test/pipeline_multi_dim_rag.py`（主入口）
- `test/benchmark_multi_dimension_rag.py`（兼容入口，内部转调 `pipeline_multi_dim_rag.py`）
- `test/benchmark_corpus_rag.sh`
- `test/benchmark_magiclens.sh`
- `test/E8_multi_dim_sweep.sh`

这意味着当前仓库是一个“模块化迁移进行中”的状态：

- `src/mrag/` 是长期保留的核心实现
- `test/*.py` 是当前可直接运行的实验驱动
- 二者之间仍有一定重复逻辑，但主要实验脚本已经开始在文件底部显式 rebind 到 `src/mrag/*`

### 1.3 当前最重要的实验主线

从代码和文档同时看，当前最重要的不是旧的 GT-rerank 线，而是下面三条：

1. `E3`: `CLIP` 直接做完整 corpus 检索
2. `E6`: `CLIP coarse retrieval -> MagicLens rerank`
3. `E8 / D01-D12`: 多维度 query decomposition + MagicLens 多路检索 + fusion

如果你现在只关心“这个项目下一步该往哪走”，优先看 `E3/E6/E8` 即可。

### 1.4 Gemma 4 基线脚本已可用

- 脚本：`test/gemma4.py`
- 功能：下载 / 加载 `gemma-4-E2B(-it)`、文本与图文 smoke test、显存日志、TTFT 与 token/s
- 图像探测：优先 `--image`，否则尝试 `CORPUS_DIR`、`data/image_corpus`、`data/`，最后回退内置合成图
- 目录自检：`python scripts/inspect_data_layout.py`，单图路径：`python scripts/inspect_data_layout.py --print-one-corpus-image`

## 2. 当前论文状态

### 2.1 论文主文档以 `paper/content.tex` 为准

当前论文已经不再只是中期答辩材料的延伸，而是在向毕业论文定稿结构收拢。最应该作为正文主入口阅读的是：

- `paper/content.tex`

它已经明确写出了当前主叙事：

- 研究对象是 `vision-centric mRAG`
- 方法主线是 `CLIP 粗召回 + MagicLens 混合检索`
- 当前最重要的失败模式是 `MagicLens direct retrieval` 的 prototype bias
- 下一阶段方法扩展是 `Question -> Query` 改写、`多维度分解`、`LoRA/蒸馏`

### 2.2 论文分析配套文档是真实有效的

当前 `paper/` 下最值得看的辅助文档是：

- `paper/README_QUICK_START.md`
- `paper/DETAILED_RESULTS.md`
- `paper/DATASET_EVALUATION_REPORT.md`
- `paper/DATA_DRIVEN_ANALYSIS.md`
- `paper/EXPERIMENTAL_SCRIPTS_GUIDE.md`

需要注意：

- 旧版 `paper/README.md` 曾引用 `PROGRESS_AND_RESULTS.md` 等当前仓库中并不存在的文件
- 因此论文入口不能再按旧 README 的顺序理解，而应该以上面这些真实文件为准

### 2.3 当前论文写作真正卡点

从 `paper/content.tex` 与 `paper/README.md` 交叉看，论文目前不是“没结构”，而是“结构已经有了，但还有几块硬结果和训练路线没补完”：

- 多维度检索路线是否能稳定超过 `E3/E6`
- 时间代价与效率对比还需要补齐
- 大模型生成训练数据与小模型 LoRA/蒸馏还没有形成完整结果
- InfoSeek 目前更适合作为扩展验证，而不是已经完成的主结果

## 3. 当前 docs 状态

### 3.1 目前最有价值的文档

如果只读 4 份文档，就读这 4 个：

1. `doc/CURRENT_STATUS_2026-04.md`
2. `doc/DATA_LAYOUT.md`
3. `doc/04-02/magiclens_vs_clip_query_image_analysis_2026-04-02.md`
4. `paper/content.tex`

它们分别回答四个不同问题：

- 现在项目整体处于什么状态
- 数据集 / 语料 / HF 缓存在机器上的真实布局
- 为什么 `MagicLens direct retrieval` 目前不如 `CLIP`
- 这些工作在论文里如何被组织成完整叙事

### 3.2 哪些文档偏旧

以下文档仍有参考价值，但不能直接当成“当前状态说明”：

- `README/PROJECT.md`
- `README/ARCHITECTURE.md`
- `README/DECISIONS.md`

原因不是它们没用，而是它们更多反映了：

- 2 月份左右的启动阶段
- 旧版同步逻辑或旧的服务器默认假设
- “MagicLens 还在接入/待测试”的阶段性判断

当前阶段，它们更适合作为“决策历史”和“工作流背景”，而不是最新状态页。

## 4. 当前服务器同步脚本状态

### 4.1 以 `module/Makefile` 为准

服务器同步、远程执行、回拉结果的主入口是：

- `Makefile`
- `module/Makefile`
- `module/.sync_ssh`
- `module/.exclude`
- `module/.alias`
- `module/pull_list.txt`
- `module/result.txt`
- `module/scripts/read_sync_value.awk`
- `module/scripts/build_rsync_filter.awk`

其中：

- 根目录 `Makefile` 只负责 `include module/Makefile`
- 真正的 sync 实现和配置都在 `module/`

这里的机制已经比较完整，不再只是“简单 rsync 一把推上去”。

### 4.2 当前推荐工作流

当前应该按下面的方式理解同步流程：

1. `make config`
2. `make sync` 预览本地到远端变更
3. `make sync y` 真正同步代码
4. `make cmd CMD='...'` 或 `mc ...` 在远端执行命令
5. `make pull` / `make pull y` 按 `pull_list.txt` 拉回代码或日志
6. `make pull result y` 或 `mr` 拉回结果产物

### 4.3 当前需要特别记住的风险

- `make sync y` 底层使用 `rsync --delete`，所以它不是“纯追加同步”
- `make pull` 现在是“清单驱动”，默认清单位于 `module/pull_list.txt`
- `log/*` 在 pull 时有保护逻辑：本地已有文件不会直接覆盖，而会生成冲突副本
- `make config` 有实际副作用，会改本地 `~/.zshrc` 并执行 `ssh-copy-id`

### 4.4 当前文档缺口

同步机制本身已经比旧文档更先进，但文档没有完全跟上，主要缺口是：

- `README/ARCHITECTURE.md` 仍在描述旧版简化 rsync 逻辑
- 旧文档仍把 `.sync_ssh/.exclude/.alias` 描述为根目录文件，而它们现在已经迁入 `module/`
- `pull_list.txt` / `result.txt` 的新位置已经是 `module/`，但旧说明还没完全改完
- 多主机 `SYNC_HOST` 的使用方式在旧文档里讲得不够完整

## 5. InfoSeek / 第二数据集状态

当前第二数据集并不是“已经完全落地”，而是“已经进入准备状态”。

可以看到两个信号：

- `paper/` 与 `doc/周报2026-03-25.md` 中已经把 `InfoSeek` 作为最优扩展方向
- `scripts/download_infoseek_images.py` 已经开始为 `InfoSeek + OVEN image_downloads` 做图像收集准备

这说明仓库当前对 InfoSeek 的真实状态是：

- 方法上已经选定为优先扩展方向
- 数据准备脚本已经开始落地
- 但它还没有成为主实验线，也还没有替代 MRAG-Bench 成为论文主体

## 6. 当前阶段你最应该知道的文档

按“今天继续做事”的优先级排序，建议如下：

### A. 第一优先级：必须知道

- `README.md`
- `doc/CURRENT_STATUS_2026-04.md`
- `doc/04-02/magiclens_vs_clip_query_image_analysis_2026-04-02.md`
- `paper/content.tex`

### B. 第二优先级：要开始做实验时再看

- `test/benchmark_corpus_rag.py`
- `test/pipeline_multi_dim_rag.py`
- `paper/EXPERIMENTAL_SCRIPTS_GUIDE.md`
- `Makefile`

### C. 第三优先级：要处理服务器/迁移时再看

- `module/README.md`
- `README/PROJECT.md`
- `README/ARCHITECTURE.md`
- `README/DECISIONS.md`
- `scripts/download_infoseek_images.py`

## 7. 当前阶段建议统一的项目表述

如果你后面要和导师、同学、AI 或自己未来回看仓库时保持口径一致，建议统一描述成：

> 当前项目以 MRAG-Bench 为核心评测基准，围绕 vision-centric mRAG 场景，研究 `CLIP 粗召回 + MagicLens 关系感知检索/重排` 的混合检索路线。现阶段已完成 E0--E7 的基线与失败模式分析，发现 MagicLens direct retrieval 存在明显 prototype bias，因此下一步重点推进多维度 query decomposition（E8 / D01-D12）、时间代价分析，以及面向 LoRA/蒸馏的小模型适配路线；同时以 InfoSeek 作为优先扩展数据集方向，并继续通过本地开发 + 远端运行的 Makefile 同步机制支撑实验工作流。

## 8. 当前最需要补同步的地方

如果后面继续整理文档，优先补这三件事：

1. 把根目录 `README.md` 精简成“当前状态 + 文档导航 + 运行入口”
2. 把 `README/ARCHITECTURE.md` 更新为当前真实的 `Makefile pull/sync/log-protect` 机制
3. 给 `paper/` 保留一个只引用真实存在文件的索引页，避免旧文件名继续误导
