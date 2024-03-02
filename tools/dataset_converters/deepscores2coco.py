import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepScoresV2 to COCO format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="root directory of DeepScoresV2 annotations",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="directory to save coco formatted label file",
    )
    return parser.parse_args()


def get_xywh_bbox(bbox: list[int]) -> list[int]:
    x_min, y_min, x_max, y_max = bbox
    return [x_min, x_max, x_max - x_min, y_max - y_min]


def convert_deepscoresv2(ann_file: Path, out_file: Path) -> None:
    with open(ann_file, encoding="utf-8") as file:
        ds_annotations = json.load(file)

    images = [
        {
            "id": int(image["id"]),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "file_name": image["filename"],
        }
        for image in ds_annotations["images"]
    ]

    annotations = [
        {
            "id": int(ann_id),
            "image_id": int(annotation["img_id"]),
            "category_id": int(annotation["cat_id"][0]),
            "area": float(annotation["area"]),
            "bbox": get_xywh_bbox(annotation["a_bbox"]),
            "iscrowd": 0,
        }
        for ann_id, annotation in ds_annotations["annotations"].items()
    ]

    categories = [
        {
            "id": int(cat_id),
            "name": categorie["name"],
        }
        for cat_id, categorie in ds_annotations["categories"].items()
        if categorie["annotation_set"] == "deepscores"
    ]

    coco_annotations = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(out_file, "w", encoding="utf-8") as file:
        json.dump(coco_annotations, file)
        print(f"Write COCO annotations to {out_file}")


def main() -> None:
    args = parse_args()
    for sub_set in ("test", "train"):
        convert_deepscoresv2(
            ann_file=args.input / f"deepscores_{sub_set}.json",
            out_file=args.output / f"annotation_coco_{sub_set}.json",
        )


if __name__ == "__main__":
    main()
