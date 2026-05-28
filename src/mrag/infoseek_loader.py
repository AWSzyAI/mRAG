"""
 * [INPUT]: 依赖 pathlib.Path 的文件系统操作、json 的行式 JSON 解析、PIL.Image 的图像加载
 * [OUTPUT]: 对外提供 InfoSeekDataset 类、iter_infoseek_records 生成器、InfaSeekSplit 枚举
 * [POS]: src/mrag 的 InfoSeek 数据加载适配层，与 mrag_bench.py 并列，为 benchmark_infoseek.py 消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 src/mrag/CLAUDE.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generator, Optional

from PIL import Image


class InfoSeekSplit(str, Enum):
    """InfoSeek 数据集分割（基于 Entity 维度）。"""

    ENTITY_TEST = "entity_test"
    ENTITY_TRAIN = "entity_train"
    ENTITY_VAL = "entity_val"
    HUMAN = "human"
    # Query 单独处理（无 image_id，用作知识增强）
    # QUERY_TRAIN = "query_train"
    # QUERY_VAL = "query_val"


@dataclass
class InfoSeekRecord:
    """单条 InfoSeek 记录的统一表示。

    Attributes:
        data_id: 记录唯一标识
        image_id: OVEN 中的图像 ID（对应 images/all/{image_id}.jpg）
        question: 开放式问题
        split: 所属数据集分割
        query_image_path: 加载后的图像路径（None 表示文件缺失）
        query_image: 加载后的 PIL Image 对象（可选，延迟加载）
    """

    data_id: str
    image_id: str
    question: str
    split: str
    query_image_path: Optional[Path] = None
    query_image: Optional[Image.Image] = None


class InfoSeekDataset:
    """InfoSeek 数据集加载器。

    提供统一的接口加载 Entity、Human 等数据分割。图像可选延迟加载。

    Usage:
        >>> ds = InfoSeekDataset(root_path="/mnt/d/mRAG/data/infoseek")
        >>> for record in ds.iter_split("entity_test", load_image=True):
        >>>     print(record.data_id, record.question)
        >>>     img = record.query_image  # PIL.Image
    """

    def __init__(self, root_path: str | Path):
        """初始化 InfoSeek 加载器。

        Args:
            root_path: InfoSeek 数据根目录（包含 Entity/, Human/, Query/, images/all/）
        """
        self.root = Path(root_path).resolve()
        self.entity_dir = self.root / "Entity"
        self.human_dir = self.root / "Human"
        self.query_dir = self.root / "Query"
        self.images_dir = self.root / "images" / "all"

    def iter_split(
        self,
        split: str | InfoSeekSplit,
        load_image: bool = False,
        skip_missing: bool = True,
    ) -> Generator[InfoSeekRecord, None, None]:
        """遍历指定数据分割。

        Args:
            split: 数据分割名称（'entity_test', 'entity_train', 'entity_val', 'human'）
            load_image: 是否加载 PIL Image（True=延迟加载；False=仅返回路径）
            skip_missing: 是否跳过图像缺失的记录（True=跳过；False=保留但 image_path=None）

        Yields:
            InfoSeekRecord: 包含元数据和可选图像的记录

        Raises:
            ValueError: 如果 split 无效或源文件缺失
        """
        if isinstance(split, InfoSeekSplit):
            split_str = split.value
        else:
            split_str = str(split).lower()

        # 映射分割到文件路径
        split_to_jsonl = {
            "entity_test": self.entity_dir / "infoseek_test.jsonl",
            "entity_train": self.entity_dir / "infoseek_train.jsonl",
            "entity_val": self.entity_dir / "infoseek_val.jsonl",
            "human": self.human_dir / "infoseek_human.jsonl",
        }

        jsonl_file = split_to_jsonl.get(split_str)
        if jsonl_file is None or not jsonl_file.exists():
            raise ValueError(
                f"Invalid split '{split_str}' or file not found: {jsonl_file}. "
                f"Valid splits: {list(split_to_jsonl.keys())}"
            )

        missing_count = 0
        with open(jsonl_file) as fp:
            for line_idx, line in enumerate(fp):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] {jsonl_file}:{line_idx} invalid JSON: {e}")
                    continue

                # 提取核心字段
                data_id = obj.get("data_id")
                image_id = obj.get("image_id")
                question = obj.get("question")

                if not (data_id and image_id and question):
                    print(
                        f"[WARN] {jsonl_file}:{line_idx} missing fields. "
                        f"Got: data_id={data_id}, image_id={image_id}, question={question}"
                    )
                    continue

                # 查找图像
                image_path = self._resolve_image_path(image_id)
                if not image_path:
                    missing_count += 1
                    if skip_missing:
                        continue
                    # 若 skip_missing=False，保留记录但 image_path=None
                    record = InfoSeekRecord(
                        data_id=data_id,
                        image_id=image_id,
                        question=question,
                        split=split_str,
                        query_image_path=None,
                        query_image=None,
                    )
                    yield record
                    continue

                # 构造记录
                record = InfoSeekRecord(
                    data_id=data_id,
                    image_id=image_id,
                    question=question,
                    split=split_str,
                    query_image_path=image_path,
                    query_image=None,
                )

                # 延迟加载图像
                if load_image:
                    try:
                        record.query_image = Image.open(image_path).convert("RGB")
                    except Exception as e:
                        print(f"[WARN] Failed to load image {image_path}: {e}")
                        if skip_missing:
                            continue
                        # 保留记录，image=None

                yield record

        if missing_count > 0:
            print(f"[INFO] {jsonl_file}: skipped {missing_count} records with missing images")

    def _resolve_image_path(self, image_id: str) -> Optional[Path]:
        """根据 image_id 查找图像文件。

        当前支持后缀：.jpg, .jpeg, .png, .JPG, .JPEG, .PNG

        Args:
            image_id: OVEN 中的图像 ID

        Returns:
            图像文件路径（若存在），否则 None
        """
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            candidate = self.images_dir / f"{image_id}{ext}"
            if candidate.is_file():
                return candidate
        return None

    def get_stats(self) -> dict:
        """统计各数据分割的记录数和缺失图像数。

        Returns:
            {
                'entity_test': {'total': N, 'with_image': N, 'missing': 0},
                'entity_train': {...},
                ...
            }
        """
        stats = {}
        for split_name in [
            "entity_test",
            "entity_train",
            "entity_val",
            "human",
        ]:
            total = 0
            with_image = 0
            for record in self.iter_split(split_name, load_image=False, skip_missing=False):
                total += 1
                if record.query_image_path:
                    with_image += 1
            stats[split_name] = {"total": total, "with_image": with_image, "missing": total - with_image}
        return stats


def iter_infoseek_records(
    root_path: str | Path,
    split: str | InfoSeekSplit = "entity_test",
    load_image: bool = False,
) -> Generator[InfoSeekRecord, None, None]:
    """便利函数：快速迭代 InfoSeek 记录。

    等价于 InfoSeekDataset(root_path).iter_split(split, load_image)。

    Args:
        root_path: InfoSeek 数据根目录
        split: 数据分割名称
        load_image: 是否加载 PIL Image

    Yields:
        InfoSeekRecord
    """
    ds = InfoSeekDataset(root_path)
    yield from ds.iter_split(split, load_image=load_image)


if __name__ == "__main__":
    # 快速验证脚本
    from pathlib import Path

    root = Path("/mnt/d/mRAG/data/infoseek")
    ds = InfoSeekDataset(root)

    print("=== InfoSeek Dataset Statistics ===")
    stats = ds.get_stats()
    for split_name, counts in stats.items():
        print(
            f"{split_name:20} total={counts['total']:7} with_image={counts['with_image']:7} "
            f"missing={counts['missing']:7}"
        )

    print("\n=== Sample Records (entity_test, first 3) ===")
    for i, record in enumerate(ds.iter_split("entity_test", load_image=True)):
        if i >= 3:
            break
        print(f"\n[{i}] {record.data_id}")
        print(f"    image_id: {record.image_id}")
        print(f"    image_path: {record.query_image_path}")
        print(f"    question: {record.question}")
        if record.query_image:
            print(f"    image_shape: {record.query_image.size}")
