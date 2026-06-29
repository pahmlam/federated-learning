import csv
import json
import zipfile

import pytest
from PIL import Image

from scripts.export_detection_subset import export_detection_subsets


def _make_export_fixture(tmp_path):
    root = tmp_path / "root"
    (root / "images").mkdir(parents=True)
    (root / "voc_labels").mkdir()
    rows = []
    for site in ("site-a", "site-b"):
        for idx, split in enumerate(("train", "val")):
            stem = f"{site}-{idx}"
            Image.new("RGB", (32, 32), color=(idx * 50, 20, 30)).save(
                root / "images" / f"{stem}.png"
            )
            (root / "voc_labels" / f"{stem}.xml").write_text(
                "<annotation><object><name>helmet</name><bndbox>"
                "<xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax>"
                "</bndbox></object></annotation>",
                encoding="utf-8",
            )
            rows.append(
                {
                    "sample_id": f"{site}_{split}_{stem}",
                    "image_path": f"images/{stem}.png",
                    "voc_path": f"voc_labels/{stem}.xml",
                    "client_id": site,
                    "split": split,
                }
            )

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root, manifest


def test_export_detection_subsets_copies_relative_manifest_and_zip(tmp_path):
    root, manifest = _make_export_fixture(tmp_path)
    output_dir = tmp_path / "exports"

    summary = export_detection_subsets(manifest, root, output_dir)

    assert summary["total"] == 4
    assert summary["sites"]["site-a"]["train"] == 1
    site_manifest = output_dir / "site-a" / "manifest.csv"
    assert site_manifest.is_file()
    assert (output_dir / "site-a" / "images").is_dir()
    assert (output_dir / "site-a" / "voc_labels").is_dir()
    assert (output_dir / "site-a.zip").is_file()

    with site_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["client_id"] for row in rows} == {"site-a"}
    assert all(row["image_path"].startswith("images/") for row in rows)
    assert all(row["voc_path"].startswith("voc_labels/") for row in rows)

    with zipfile.ZipFile(output_dir / "site-a.zip") as archive:
        names = set(archive.namelist())
    assert "manifest.csv" in names
    assert any(name.startswith("images/") for name in names)
    assert any(name.startswith("voc_labels/") for name in names)

    summary_file = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_file["sites"]["site-b"]["val"] == 1


def test_export_detection_subsets_refuses_existing_targets(tmp_path):
    root, manifest = _make_export_fixture(tmp_path)
    output_dir = tmp_path / "exports"
    export_detection_subsets(manifest, root, output_dir)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_detection_subsets(manifest, root, output_dir)

    summary = export_detection_subsets(manifest, root, output_dir, overwrite=True)
    assert summary["sites"]["site-a"]["total"] == 2
