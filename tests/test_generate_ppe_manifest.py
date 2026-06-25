"""Tests for the reproducible PPE manifest generator."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.data.manifest_generator import (
    DEFAULT_CORE_PPE,
    SAFE_LABEL,
    UNSAFE_LABEL,
    ManifestRow,
    classify_objects,
    collect_labeled_samples,
    generate_manifest_rows,
    summarize_rows,
    write_manifest,
)
from src.data.real_data import load_manifest


def _write_voc(path: Path, object_names: list[str]) -> None:
    objects = "".join(
        f"<object><name>{name}</name></object>" for name in object_names
    )
    path.write_text(f"<annotation>{objects}</annotation>", encoding="utf-8")


def _build_fixture(tmp_path: Path, n_safe: int = 12, n_unsafe: int = 12) -> tuple[Path, Path]:
    """Create a tiny VOC + images fixture with mixed extensions."""

    voc_dir = tmp_path / "voc_labels"
    images_dir = tmp_path / "images"
    voc_dir.mkdir()
    images_dir.mkdir()

    for i in range(n_safe):
        stem = f"safe_{i:03d}"
        # Alternate extension so we exercise .jpg/.jpeg resolution.
        ext = ".jpg" if i % 2 == 0 else ".jpeg"
        _write_voc(voc_dir / f"{stem}.xml", ["person", "helmet"])
        (images_dir / f"{stem}{ext}").write_bytes(b"x")
    for i in range(n_unsafe):
        stem = f"unsafe_{i:03d}"
        ext = ".jpeg" if i % 2 == 0 else ".jpg"
        _write_voc(voc_dir / f"{stem}.xml", ["person", "head"])
        (images_dir / f"{stem}{ext}").write_bytes(b"x")

    # Annotation without a matching image must be skipped.
    _write_voc(voc_dir / "orphan.xml", ["helmet"])
    return voc_dir, images_dir


def test_classify_objects_proxy_rule():
    assert classify_objects(["person", "helmet"], DEFAULT_CORE_PPE) == SAFE_LABEL
    assert classify_objects(["safety-vest"], DEFAULT_CORE_PPE) == SAFE_LABEL
    assert classify_objects(["person", "head", "tools"], DEFAULT_CORE_PPE) == UNSAFE_LABEL
    assert classify_objects([], DEFAULT_CORE_PPE) == UNSAFE_LABEL


def test_classify_objects_is_case_insensitive_and_trims():
    assert classify_objects([" Helmet "], DEFAULT_CORE_PPE) == SAFE_LABEL


def test_collect_labeled_samples_resolves_extensions_and_skips_orphans(tmp_path):
    voc_dir, images_dir = _build_fixture(tmp_path, n_safe=3, n_unsafe=3)

    samples = collect_labeled_samples(voc_dir, images_dir)

    # 6 real images, orphan.xml skipped (no image).
    assert len(samples) == 6
    labels = Counter(s.label for s in samples)
    assert labels[SAFE_LABEL] == 3
    assert labels[UNSAFE_LABEL] == 3
    # Mixed extensions are resolved to the actual file on disk.
    extensions = {Path(s.filename).suffix for s in samples}
    assert extensions <= {".jpg", ".jpeg"}
    assert all((images_dir / s.filename).exists() for s in samples)


def _gen(tmp_path: Path):
    voc_dir, images_dir = _build_fixture(tmp_path)
    samples = collect_labeled_samples(voc_dir, images_dir)
    return generate_manifest_rows(
        samples,
        sites=["site-a", "site-b"],
        safe_ratios=[0.75, 0.25],
        per_site=8,
        val_fraction=0.25,
        seed=2026,
    )


def test_generate_manifest_rows_distribution(tmp_path):
    rows = _gen(tmp_path)

    assert len(rows) == 16  # 2 sites x 8 samples
    by_site = Counter((r.client_id, r.label) for r in rows)
    # site-a skews safe (0.75 -> 6 safe / 2 unsafe), site-b mirrors it.
    assert by_site[("site-a", SAFE_LABEL)] == 6
    assert by_site[("site-a", UNSAFE_LABEL)] == 2
    assert by_site[("site-b", SAFE_LABEL)] == 2
    assert by_site[("site-b", UNSAFE_LABEL)] == 6


def test_generate_manifest_rows_each_site_has_train_val_and_both_classes(tmp_path):
    rows = _gen(tmp_path)

    for site in ("site-a", "site-b"):
        site_rows = [r for r in rows if r.client_id == site]
        splits = {r.split for r in site_rows}
        classes = {r.label for r in site_rows}
        assert splits == {"train", "val"}
        assert classes == {SAFE_LABEL, UNSAFE_LABEL}


def test_generate_manifest_rows_no_leakage(tmp_path):
    rows = _gen(tmp_path)

    # No underlying image file is reused across sites/splits.
    image_paths = [r.image_path for r in rows]
    assert len(image_paths) == len(set(image_paths))
    # sample_ids are unique too.
    assert len({r.sample_id for r in rows}) == len(rows)


def test_generate_manifest_rows_is_deterministic(tmp_path):
    voc_dir, images_dir = _build_fixture(tmp_path)
    samples = collect_labeled_samples(voc_dir, images_dir)
    kwargs = dict(
        sites=["site-a", "site-b"],
        safe_ratios=[0.75, 0.25],
        per_site=8,
        val_fraction=0.25,
        seed=2026,
    )
    rows_a = generate_manifest_rows(samples, **kwargs)
    rows_b = generate_manifest_rows(samples, **kwargs)
    assert rows_a == rows_b


def test_generate_manifest_rows_seed_changes_selection(tmp_path):
    voc_dir, images_dir = _build_fixture(tmp_path)
    samples = collect_labeled_samples(voc_dir, images_dir)
    kwargs = dict(
        sites=["site-a", "site-b"],
        safe_ratios=[0.75, 0.25],
        per_site=8,
        val_fraction=0.25,
    )
    rows_seed_a = generate_manifest_rows(samples, seed=1, **kwargs)
    rows_seed_b = generate_manifest_rows(samples, seed=999, **kwargs)
    paths_a = {r.image_path for r in rows_seed_a}
    paths_b = {r.image_path for r in rows_seed_b}
    assert paths_a != paths_b


def test_generate_manifest_rows_rejects_mismatched_ratios(tmp_path):
    voc_dir, images_dir = _build_fixture(tmp_path)
    samples = collect_labeled_samples(voc_dir, images_dir)
    with pytest.raises(ValueError, match="sites and safe_ratios"):
        generate_manifest_rows(
            samples,
            sites=["site-a", "site-b"],
            safe_ratios=[0.5],
            per_site=8,
            val_fraction=0.25,
            seed=2026,
        )


def test_generate_manifest_rows_rejects_insufficient_pool(tmp_path):
    voc_dir, images_dir = _build_fixture(tmp_path, n_safe=2, n_unsafe=2)
    samples = collect_labeled_samples(voc_dir, images_dir)
    with pytest.raises(ValueError, match="not enough"):
        generate_manifest_rows(
            samples,
            sites=["site-a", "site-b"],
            safe_ratios=[0.75, 0.25],
            per_site=8,
            val_fraction=0.25,
            seed=2026,
        )


def test_write_manifest_roundtrips_through_loader(tmp_path):
    voc_dir, images_dir = _build_fixture(tmp_path)
    samples = collect_labeled_samples(voc_dir, images_dir)
    rows = generate_manifest_rows(
        samples,
        sites=["site-a", "site-b"],
        safe_ratios=[0.75, 0.25],
        per_site=8,
        val_fraction=0.25,
        seed=2026,
    )
    output = tmp_path / "manifest.csv"
    write_manifest(rows, output)

    # The official loader validates schema, unique sample_ids and file presence.
    loaded = load_manifest(output, root_dir=tmp_path, require_files=True)
    assert len(loaded) == 16
    assert {record.client_id for record in loaded} == {"site-a", "site-b"}


def test_summarize_rows_counts(tmp_path):
    rows = _gen(tmp_path)
    summary = summarize_rows(rows)
    assert summary["total"] == 16
    assert summary["per_site"]["site-a"]["total"] == 8
    assert summary["per_site"]["site-a"][SAFE_LABEL] == 6


def test_manifest_row_is_immutable():
    row = ManifestRow("id", "images/x.jpg", "safe", "site-a", "train")
    with pytest.raises(Exception):
        row.label = "unsafe"  # type: ignore[misc]
