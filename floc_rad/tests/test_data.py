from pathlib import Path

from flocrad.data import build_correction_dataset, generate_synthetic_manifest, make_composition_splits, read_jsonl


def test_synthetic_prepare(tmp_path: Path):
    manifest = generate_synthetic_manifest(str(tmp_path / "data"), n=40, seed=7)
    all_examples = tmp_path / "splits" / "all.jsonl"
    stats = build_correction_dataset(
        manifest,
        str(all_examples),
        examples_per_study=2,
        error_counts=(1, 2),
        clean_fraction=0.1,
        seed=7,
    )
    assert sum(stats.values()) > 0
    metadata = make_composition_splits(
        str(all_examples),
        str(tmp_path / "splits"),
        holdout_fraction=0.3,
        seed=7,
        min_compound_count=1,
    )
    assert (tmp_path / "splits" / "train.jsonl").exists()
    assert read_jsonl(str(tmp_path / "splits" / "train.jsonl"))
    assert "counts" in metadata
