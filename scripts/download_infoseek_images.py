#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from pathlib import Path
from urllib.request import urlretrieve


MAPPING_URL = "http://storage.googleapis.com/gresearch/open-vision-language/ovenid2impath.csv"


def iter_json_or_jsonl(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for item in obj:
            yield item
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")


def load_entity_dataid_to_imageid(entity_dir: Path) -> dict:
    mapping = {}
    for fp in sorted(entity_dir.glob("*.json*")):
        for item in iter_json_or_jsonl(fp):
            data_id = item.get("data_id")
            image_id = item.get("image_id")
            if data_id and image_id:
                mapping[data_id] = image_id
    return mapping


def collect_subset_image_ids(infoseek_root: Path, max_images_per_subset: int | None):
    subset_to_ids = {"Entity": [], "Human": [], "Query": []}
    entity_map = load_entity_dataid_to_imageid(infoseek_root / "Entity")

    for subset in ("Entity", "Human"):
        subset_dir = infoseek_root / subset
        for fp in sorted(subset_dir.glob("*.json*")):
            for item in iter_json_or_jsonl(fp):
                image_id = item.get("image_id")
                if image_id:
                    subset_to_ids[subset].append(image_id)

    query_dir = infoseek_root / "Query"
    for fp in sorted(query_dir.glob("*.json*")):
        for item in iter_json_or_jsonl(fp):
            data_id = item.get("data_id")
            if not data_id:
                continue
            image_id = entity_map.get(data_id)
            if image_id:
                subset_to_ids["Query"].append(image_id)

    for subset, ids in subset_to_ids.items():
        deduped = list(dict.fromkeys(ids))
        if max_images_per_subset is not None:
            deduped = deduped[:max_images_per_subset]
        subset_to_ids[subset] = deduped

    return subset_to_ids


def ensure_mapping_csv(mapping_csv: Path):
    if mapping_csv.exists():
        return
    mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading mapping CSV -> {mapping_csv}")
    urlretrieve(MAPPING_URL, mapping_csv)


def load_needed_relpaths(mapping_csv: Path, needed_ids: set[str]):
    result = {}
    with mapping_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue
            oven_id, relpath = row
            if oven_id in needed_ids:
                result[oven_id] = relpath
    return result


def source_script_for_prefix(prefix: str) -> str | None:
    table = {
        "aircraft": "download_aircraft.sh",
        "car196": "download_car196.sh",
        "coco": "download_coco.sh",
        "food101": "download_food101.sh",
        "gldv2": "download_gldv2.sh",
        "imagenet21k": "download_imagenet.sh",
        "inat": "download_inat.sh",
        "oxfordflower": "download_oxfordflower.sh",
        "sports100": "download_sports100.sh",
        "sun397": "download_sun397.sh",
        "textvqa": "download_textvqa.sh",
        "v7w": "download_v7w.sh",
        "vg": "download_vg.sh",
    }
    return table.get(prefix)


def copy_images(
    subset_to_ids: dict,
    id_to_relpath: dict,
    image_downloads_root: Path,
    infoseek_root: Path,
    dry_run: bool,
):
    total_copied = 0
    for subset, ids in subset_to_ids.items():
        out_dir = infoseek_root / subset / "images"
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        missing_mapping = 0
        missing_source_file = 0

        for oven_id in ids:
            relpath = id_to_relpath.get(oven_id)
            if not relpath:
                missing_mapping += 1
                continue

            src = image_downloads_root / relpath
            if not src.exists():
                missing_source_file += 1
                continue

            dst = out_dir / f"{oven_id}{src.suffix}"
            if dry_run:
                copied += 1
                continue

            if not dst.exists():
                shutil.copyfile(src, dst)
                copied += 1

        total_copied += copied
        print(
            f"[SUMMARY] {subset}: selected={len(ids)} copied={copied} "
            f"missing_mapping={missing_mapping} missing_source={missing_source_file}"
        )

    return total_copied


def main():
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Collect image_id from data/infoseek and copy only matched OVEN images to "
            "data/infoseek/{Entity,Human,Query}/images"
        )
    )
    parser.add_argument(
        "--infoseek-root",
        type=Path,
        default=repo_root / "data" / "infoseek",
        help="Path to data/infoseek",
    )
    parser.add_argument(
        "--image-downloads-root",
        type=Path,
        default=repo_root / "github" / "oven" / "image_downloads",
        help="Path to github/oven/image_downloads",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=repo_root / "github" / "oven" / "image_downloads" / "ovenid2impath.csv",
        help="Path to ovenid2impath.csv",
    )
    parser.add_argument(
        "--max-images-per-subset",
        type=int,
        default=None,
        help="Only keep first N unique image_id for each subset (Entity/Human/Query), useful for small-scale tests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not copy files, only print statistics and required source scripts",
    )

    args = parser.parse_args()

    subset_to_ids = collect_subset_image_ids(args.infoseek_root, args.max_images_per_subset)
    needed_ids = set().union(*subset_to_ids.values())
    print(f"[INFO] Unique needed image_id: {len(needed_ids)}")

    ensure_mapping_csv(args.mapping_csv)
    id_to_relpath = load_needed_relpaths(args.mapping_csv, needed_ids)

    source_prefixes = sorted({p.split("/", 1)[0] for p in id_to_relpath.values()})
    print("[INFO] Required source folders:")
    for prefix in source_prefixes:
        script = source_script_for_prefix(prefix)
        if script:
            print(f"  - {prefix}: run github/oven/image_downloads/{script}")
        else:
            print(f"  - {prefix}: no mapped script name, check manually")

    unresolved = len(needed_ids) - len(id_to_relpath)
    if unresolved:
        print(f"[WARN] image_id not found in mapping CSV: {unresolved}")

    copied = copy_images(
        subset_to_ids=subset_to_ids,
        id_to_relpath=id_to_relpath,
        image_downloads_root=args.image_downloads_root,
        infoseek_root=args.infoseek_root,
        dry_run=args.dry_run,
    )

    mode = "DRY-RUN" if args.dry_run else "COPY"
    print(f"[DONE] mode={mode}, total_effective={copied}")


if __name__ == "__main__":
    main()
