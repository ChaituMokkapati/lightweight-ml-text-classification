"""Build Overleaf and Zenodo release zip archives with consistent naming."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
STAGING = ROOT / "_release_staging"

OVERLEAF_ZIP = ROOT / (
    "Mokkapati-Syyed-Devarasetty-comparative-study-lightweight-ml-text-classification-overleaf.zip"
)
ZENODO_ZIP = ROOT / "Mokkapati-Syyed-Devarasetty-lightweight-ml-text-classification-v1.1.0.zip"


def _zip_dir(src: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src).as_posix())


def build_overleaf_zip() -> Path:
    stage = STAGING / "overleaf"
    if stage.exists():
        shutil.rmtree(stage)
    fig = stage / "figures"
    fig.mkdir(parents=True)
    for name in (
        "manuscript.tex",
        "references.bib",
        "sn-jnl.cls",
        "sn-mathphys-num.bst",
    ):
        shutil.copy2(MANUSCRIPT / name, stage / name)
    for png in (MANUSCRIPT / "figures").glob("*.png"):
        shutil.copy2(png, fig / png.name)
    _zip_dir(stage, OVERLEAF_ZIP)
    return OVERLEAF_ZIP


def build_zenodo_zip() -> Path:
    stage = STAGING / "zenodo"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    for name in ("README.md", "LICENSE"):
        shutil.copy2(ROOT / name, stage / name)

    shutil.copytree(ROOT / "experiment", stage / "experiment", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(
        ROOT / "results2",
        stage / "results2",
        ignore=shutil.ignore_patterns("*.joblib", "__pycache__"),
    )

    fig = stage / "figures"
    fig.mkdir()
    for png in (MANUSCRIPT / "figures").glob("*.png"):
        shutil.copy2(png, fig / png.name)

    shutil.copy2(MANUSCRIPT / "manuscript.tex", stage / "manuscript.tex")
    shutil.copy2(MANUSCRIPT / "references.bib", stage / "references.bib")

    _zip_dir(stage, ZENODO_ZIP)
    return ZENODO_ZIP


def main() -> None:
    overleaf = build_overleaf_zip()
    zenodo = build_zenodo_zip()
    if STAGING.exists():
        shutil.rmtree(STAGING)
    print(f"Overleaf: {overleaf.name} ({overleaf.stat().st_size:,} bytes)")
    print(f"Zenodo:   {zenodo.name} ({zenodo.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
