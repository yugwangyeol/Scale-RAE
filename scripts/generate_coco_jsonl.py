"""
COCO train2017 이미지 목록으로 coco_oc_train.jsonl 생성.

사용법:
    python scripts/generate_coco_jsonl.py \
        --coco_dir /data/coco \
        --output /data/coco_oc_train.jsonl \
        [--annotation instances_train2017.json]  # 기본값: coco_dir/annotations/instances_train2017.json
        [--min_objects 1]   # n_objects가 이 수 미만인 이미지 제외 (기본 1)
        [--max_objects 50]  # n_objects가 이 수 초과인 이미지 제외 (기본 50, 이상값 제거)

출력 JSONL 형식 (한 줄 = 한 이미지):
    {"image": "000000391895.jpg", "image_id": "391895"}

scale_rae/train/spmd_trainer.py의 COCOReconstructionDataset에서 읽는 포맷과 동일.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="COCO OC JSONL 생성")
    parser.add_argument(
        "--coco_dir", type=str, required=True,
        help="COCO 루트 디렉토리 (train2017/, annotations/ 포함)",
    )
    parser.add_argument(
        "--output", type=str, default="coco_oc_train.jsonl",
        help="출력 JSONL 파일 경로",
    )
    parser.add_argument(
        "--annotation", type=str, default=None,
        help="instances_train2017.json 경로 (기본: coco_dir/annotations/instances_train2017.json)",
    )
    parser.add_argument(
        "--min_objects", type=int, default=1,
        help="최소 유효 object 수 (이 수 미만 이미지는 제외)",
    )
    parser.add_argument(
        "--max_objects", type=int, default=50,
        help="최대 유효 object 수 (이 수 초과 이미지는 제외, 이상값 방지)",
    )
    parser.add_argument(
        "--min_area", type=float, default=1024.0,
        help="annotation 최소 면적 (px²). 이 미만 annotation은 object 수에서 제외",
    )
    parser.add_argument(
        "--image_dir", type=str, default=None,
        help="이미지 폴더 (기본: coco_dir/train2017)",
    )
    return parser.parse_args()


def load_n_objects(annotation_path: str, min_area: float) -> dict:
    """image_id -> n_valid_objects 매핑 반환."""
    print(f"[1/3] COCO annotation 로드 중: {annotation_path}")
    with open(annotation_path, "r") as f:
        coco = json.load(f)

    n_objects_map = {}
    total_anns = 0
    skipped_crowd = 0
    skipped_small = 0

    for ann in coco.get("annotations", []):
        total_anns += 1
        if ann.get("iscrowd", 0):
            skipped_crowd += 1
            continue
        if ann.get("area", 0) < min_area:
            skipped_small += 1
            continue
        img_id = str(ann["image_id"])
        n_objects_map[img_id] = n_objects_map.get(img_id, 0) + 1

    print(
        f"    총 annotation: {total_anns:,}"
        f" | iscrowd 제외: {skipped_crowd:,}"
        f" | 소형 제외(<{min_area:.0f}px²): {skipped_small:,}"
    )
    print(f"    유효 이미지 수: {len(n_objects_map):,}")

    avg = sum(n_objects_map.values()) / max(len(n_objects_map), 1)
    print(f"    평균 objects/image: {avg:.1f}")

    return n_objects_map


def collect_images(image_dir: str) -> list:
    """이미지 디렉토리에서 .jpg 파일 목록 반환."""
    print(f"[2/3] 이미지 스캔 중: {image_dir}")
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in exts
    ])
    print(f"    발견된 이미지: {len(images):,}")
    return images


def parse_image_id(filename: str) -> str:
    """000000391895.jpg -> '391895'"""
    basename = Path(filename).stem
    digits = re.sub(r"^0+", "", re.sub(r"\D", "", basename))
    return digits if digits else basename


def main():
    args = parse_args()

    coco_dir    = Path(args.coco_dir)
    image_dir   = Path(args.image_dir) if args.image_dir else coco_dir / "train2017"
    ann_path    = args.annotation or str(coco_dir / "annotations" / "instances_train2017.json")
    output_path = args.output

    if not image_dir.exists():
        print(f"[ERROR] 이미지 폴더가 없습니다: {image_dir}")
        sys.exit(1)
    if not Path(ann_path).exists():
        print(f"[ERROR] annotation 파일이 없습니다: {ann_path}")
        sys.exit(1)

    # 1. n_objects 맵 구성
    n_objects_map = load_n_objects(ann_path, args.min_area)

    # 2. 이미지 목록 수집
    images = collect_images(str(image_dir))

    # 3. JSONL 출력
    print(f"[3/3] JSONL 생성 중: {output_path}")
    n_written   = 0
    n_no_ann    = 0
    n_too_few   = 0
    n_too_many  = 0

    with open(output_path, "w") as fout:
        for fname in images:
            image_id = parse_image_id(fname)
            n_obj    = n_objects_map.get(image_id, 0)

            if n_obj == 0:
                n_no_ann += 1
                continue
            if n_obj < args.min_objects:
                n_too_few += 1
                continue
            if n_obj > args.max_objects:
                n_too_many += 1
                continue

            record = {"image": fname, "image_id": image_id}
            fout.write(json.dumps(record) + "\n")
            n_written += 1

    print()
    print("=" * 50)
    print(f"  총 이미지:           {len(images):>8,}")
    print(f"  annotation 없음:     {n_no_ann:>8,}")
    print(f"  object 너무 적음:    {n_too_few:>8,} (< {args.min_objects})")
    print(f"  object 너무 많음:    {n_too_many:>8,} (> {args.max_objects})")
    print(f"  최종 학습 이미지:    {n_written:>8,}")
    print(f"  출력 파일:           {output_path}")
    print("=" * 50)

    if n_written == 0:
        print("[WARNING] 출력된 데이터가 없습니다. 경로와 필터 조건을 확인하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
