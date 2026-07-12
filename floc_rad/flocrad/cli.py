from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import (
    build_chexpert_plus_manifest,
    build_correction_dataset,
    build_mimic_manifest,
    build_openi_manifest,
    build_universal_manifest,
    download_mimic,
    download_openi,
    generate_synthetic_manifest,
    make_composition_splits,
    read_jsonl,
    stage_chexpert_plus,
)
from .model import CorrectionDataset, SimpleTokenizer, collate_batch, load_checkpoint, load_radiograph
from .schema import apply_edits, decode_edits, parse_report, realize_state
from .train import evaluate_model, predict_items, train_model


def _bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FLOC-Rad end-to-end pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Download or stage a dataset")
    download.add_argument("--dataset", choices=["openi", "mimic", "chexpert_plus", "synthetic"], required=True)
    download.add_argument("--output-dir", required=True)
    download.add_argument("--source-dir", default="")
    download.add_argument("--username", default=os.environ.get("PHYSIONET_USERNAME", ""))
    download.add_argument("--password", default=os.environ.get("PHYSIONET_PASSWORD", ""))
    download.add_argument("--max-images", type=int, default=0)
    download.add_argument("--synthetic-size", type=int, default=120)
    download.add_argument("--copy", action="store_true")

    manifest = sub.add_parser("manifest", help="Create a common image-report manifest")
    manifest.add_argument("--dataset", choices=["openi", "mimic", "chexpert_plus", "universal", "synthetic"], required=True)
    manifest.add_argument("--data-dir", required=True)
    manifest.add_argument("--output-csv", required=True)
    manifest.add_argument("--input-csv", default="")
    manifest.add_argument("--synthetic-size", type=int, default=120)
    manifest.add_argument("--seed", type=int, default=42)

    prepare = sub.add_parser("prepare", help="Inject controlled errors and create composition splits")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--examples-per-study", type=int, default=3)
    prepare.add_argument("--error-counts", default="1,2,3")
    prepare.add_argument("--clean-fraction", type=float, default=0.1)
    prepare.add_argument("--composition-unit", choices=["operator", "operator_finding", "full"], default="operator_finding")
    prepare.add_argument("--holdout-fraction", type=float, default=0.25)
    prepare.add_argument("--min-compound-count", type=int, default=4)
    prepare.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train", help="Train the multimodal edit predictor")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--image-backbone", choices=["tiny", "resnet18", "resnet50"], default="resnet18")
    train.add_argument("--image-pretrained", type=_bool, default=True)
    train.add_argument("--freeze-image", type=_bool, default=False)
    train.add_argument("--image-size", type=int, default=224)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--max-text-length", type=int, default=192)
    train.add_argument("--patience", type=int, default=3)
    train.add_argument("--seed", type=int, default=42)

    evaluate = sub.add_parser("evaluate", help="Evaluate seen/unseen composition splits")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--image-size", type=int, default=224)
    evaluate.add_argument("--num-workers", type=int, default=4)

    predict = sub.add_parser("predict", help="Correct a single report")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--report", required=True)
    predict.add_argument("--device", default="cuda")
    predict.add_argument("--threshold", type=float, default=None)

    pipeline = sub.add_parser("pipeline", help="Prepare, train, and evaluate from a manifest")
    pipeline.add_argument("--manifest", required=True)
    pipeline.add_argument("--work-dir", required=True)
    pipeline.add_argument("--device", default="cuda")
    pipeline.add_argument("--epochs", type=int, default=10)
    pipeline.add_argument("--batch-size", type=int, default=16)
    pipeline.add_argument("--image-backbone", choices=["tiny", "resnet18", "resnet50"], default="resnet18")
    pipeline.add_argument("--image-pretrained", type=_bool, default=True)
    pipeline.add_argument("--freeze-image", type=_bool, default=False)
    pipeline.add_argument("--examples-per-study", type=int, default=3)
    pipeline.add_argument("--error-counts", default="1,2,3")
    pipeline.add_argument("--clean-fraction", type=float, default=0.1)
    pipeline.add_argument("--holdout-fraction", type=float, default=0.25)
    pipeline.add_argument("--min-compound-count", type=int, default=4)
    pipeline.add_argument("--num-workers", type=int, default=4)
    pipeline.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        if args.dataset == "openi":
            download_openi(args.output_dir)
        elif args.dataset == "mimic":
            download_mimic(args.output_dir, args.username, args.password, args.max_images)
        elif args.dataset == "chexpert_plus":
            if not args.source_dir:
                raise ValueError("CheXpert Plus must first be exported from Redivis; pass --source-dir.")
            stage_chexpert_plus(args.source_dir, args.output_dir, symlink=not args.copy)
        else:
            print(generate_synthetic_manifest(args.output_dir, args.synthetic_size))
        return

    if args.command == "manifest":
        if args.dataset == "openi":
            frame = build_openi_manifest(args.data_dir, args.output_csv, args.seed)
        elif args.dataset == "mimic":
            frame = build_mimic_manifest(args.data_dir, args.output_csv)
        elif args.dataset == "chexpert_plus":
            frame = build_chexpert_plus_manifest(args.data_dir, args.output_csv, args.seed)
        elif args.dataset == "universal":
            frame = build_universal_manifest(args.data_dir, args.output_csv, args.input_csv)
        else:
            source = generate_synthetic_manifest(args.data_dir, args.synthetic_size, args.seed)
            frame = build_universal_manifest(args.data_dir, args.output_csv, source)
        print(frame.groupby(["dataset", "split"]).size())
        return

    if args.command == "prepare":
        work = Path(args.output_dir)
        all_jsonl = work / "all_examples.jsonl"
        stats = build_correction_dataset(
            args.manifest,
            str(all_jsonl),
            args.examples_per_study,
            [int(value) for value in args.error_counts.split(",") if value.strip()],
            args.clean_fraction,
            args.composition_unit,
            args.seed,
        )
        metadata = make_composition_splits(str(all_jsonl), str(work), args.holdout_fraction, args.seed, args.min_compound_count)
        print(json.dumps({"dataset": stats, "splits": metadata}, indent=2))
        return

    if args.command == "train":
        checkpoint = train_model(
            str(Path(args.data_dir) / "train.jsonl"),
            str(Path(args.data_dir) / "val.jsonl"),
            args.output_dir,
            args.device,
            args.seed,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            args.image_backbone,
            args.image_pretrained,
            args.freeze_image,
            args.image_size,
            args.num_workers,
            args.max_text_length,
            args.patience,
        )
        print(checkpoint)
        return

    if args.command == "evaluate":
        metrics = evaluate_model(
            args.checkpoint,
            str(Path(args.data_dir) / "test_seen.jsonl"),
            str(Path(args.data_dir) / "test_unseen.jsonl"),
            args.output_dir,
            args.device,
            args.batch_size,
            args.image_size,
            args.num_workers,
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "predict":
        state = parse_report(args.report)
        item = {
            "id": "single",
            "image_path": args.image,
            "corrupted_report": args.report,
            "gold_report": "",
            "corrupted_state": json.dumps({key: value.to_dict() for key, value in state.items()}),
            "gold_state": json.dumps({key: value.to_dict() for key, value in state.items()}),
            "edits": [],
            "composition": "UNKNOWN",
            "k": 0,
        }
        row = predict_items(args.checkpoint, [item], args.device, 1, 224, 0, args.threshold)[0]
        print(json.dumps({"edits": row["pred_edits"], "corrected_report": row["corrected_report"]}, indent=2))
        return

    if args.command == "pipeline":
        work = Path(args.work_dir)
        split_dir = work / "splits"
        model_dir = work / "model"
        eval_dir = work / "evaluation"
        all_jsonl = split_dir / "all_examples.jsonl"
        build_correction_dataset(
            args.manifest,
            str(all_jsonl),
            args.examples_per_study,
            [int(value) for value in args.error_counts.split(",") if value.strip()],
            args.clean_fraction,
            "operator_finding",
            args.seed,
        )
        make_composition_splits(str(all_jsonl), str(split_dir), args.holdout_fraction, args.seed, args.min_compound_count)
        checkpoint = train_model(
            str(split_dir / "train.jsonl"), str(split_dir / "val.jsonl"), str(model_dir),
            args.device, args.seed, args.epochs, args.batch_size, 2e-4, 1e-4,
            args.image_backbone, args.image_pretrained, args.freeze_image, 224, args.num_workers, 192, 3,
        )
        metrics = evaluate_model(
            checkpoint, str(split_dir / "test_seen.jsonl"), str(split_dir / "test_unseen.jsonl"),
            str(eval_dir), args.device, args.batch_size, 224, args.num_workers,
        )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
