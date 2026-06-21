"""Publish a new Zenodo version via the InvenioRDM Records API."""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "Mokkapati-Syyed-Devarasetty-lightweight-ml-text-classification-v1.1.0.zip"
ZENODO_API = "https://zenodo.org/api"
PREVIOUS_RECORD_ID = 20780973
TOKEN = os.environ.get("ZENODO_ACCESS_TOKEN", "")
VERSION = "v1.1.0"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.inveniordm.v1+json",
}


def request(
    url: str,
    data: bytes | None = None,
    method: str = "GET",
    content_type: str = "application/json",
) -> dict:
    headers = dict(HEADERS)
    if data is not None:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode()
        return json.loads(body) if body else {}


def creator(given: str, family: str, affiliation: str) -> dict:
    return {
        "person_or_org": {
            "type": "personal",
            "given_name": given,
            "family_name": family,
            "name": f"{family}, {given}",
        },
        "affiliations": [{"name": affiliation}],
    }


METADATA = {
    "title": "Lightweight ML Text Classification Benchmark (Code and Results)",
    "publisher": "Zenodo",
    "publication_date": date.today().isoformat(),
    "description": (
        "Reproducible CPU-only benchmark comparing five classical machine learning "
        "models (Logistic Regression, Linear SVM, Naive Bayes, Random Forest, XGBoost) "
        "for text classification with a unified TF-IDF pipeline on SMS Spam and AG News. "
        "Includes experiment notebook, result CSVs, figures, and manuscript source. "
        "Companion to: A Comparative Study of Classical Lightweight Machine Learning "
        "Models for Text Classification."
    ),
    "creators": [
        creator("Chaitanya", "Mokkapati", "DVR & Dr. HS MIC College of Technology"),
        creator("Nagulmeera", "Syyed", "DVR & Dr. HS MIC College of Technology"),
        creator("Prasad", "Devarasetty", "DVR & Dr. HS MIC College of Technology"),
    ],
    "keywords": [
        {"subject": "text classification"},
        {"subject": "machine learning"},
        {"subject": "TF-IDF"},
        {"subject": "benchmark"},
        {"subject": "CPU-only"},
    ],
    "resource_type": {"id": "software"},
    "version": VERSION,
    "language": {"id": "eng", "title": {"en": "English"}},
    "rights": [
        {
            "id": "mit",
            "title": {"en": "MIT License"},
            "link": "https://opensource.org/licenses/MIT",
        }
    ],
    "related_identifiers": [
        {
            "identifier": "https://github.com/ChaituMokkapati/lightweight-ml-text-classification",
            "relation_type": {"id": "issupplementto", "title": {"en": "Is supplement to"}},
            "resource_type": {"id": "software", "title": {"en": "Software"}},
            "scheme": "url",
        }
    ],
}


def main() -> int:
    if not TOKEN:
        print("Set ZENODO_ACCESS_TOKEN")
        return 1
    if not ZIP_PATH.exists():
        print(f"Missing archive: {ZIP_PATH}")
        print("Run: python experiment/build_release_archives.py")
        return 1

    try:
        draft = request(
            f"{ZENODO_API}/records/{PREVIOUS_RECORD_ID}/versions",
            data=b"{}",
            method="POST",
        )
    except urllib.error.HTTPError as exc:
        print(f"Failed to create new version: {exc.code} {exc.read().decode(errors='replace')}")
        return 1

    draft_id = draft["id"]
    draft_url = f"{ZENODO_API}/records/{draft_id}/draft"
    files_url = f"{draft_url}/files"
    print(f"Created draft {draft_id}")

    payload = json.dumps(
        {
            "access": {"record": "public", "files": "public"},
            "metadata": METADATA,
        }
    ).encode()
    request(draft_url, data=payload, method="PUT")
    print("Metadata updated")

    with ZIP_PATH.open("rb") as handle:
        file_data = handle.read()

    try:
        request(files_url, data=json.dumps([{"key": ZIP_PATH.name}]).encode(), method="POST")
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
        request(f"{files_url}/{ZIP_PATH.name}", method="DELETE")
        request(files_url, data=json.dumps([{"key": ZIP_PATH.name}]).encode(), method="POST")

    upload_req = urllib.request.Request(
        f"{files_url}/{ZIP_PATH.name}/content",
        data=file_data,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.inveniordm.v1+json",
            "Content-Type": "application/octet-stream",
        },
        method="PUT",
    )
    with urllib.request.urlopen(upload_req) as resp:
        print("Uploaded content", ZIP_PATH.name, resp.status)

    request(f"{files_url}/{ZIP_PATH.name}/commit", data=b"{}", method="POST")
    print("Committed file")

    try:
        published = request(
            f"{ZENODO_API}/records/{draft_id}/draft/actions/publish",
            data=b"{}",
            method="POST",
        )
    except urllib.error.HTTPError as exc:
        print(f"Publish failed: {exc.code} {exc.read().decode(errors='replace')}")
        return 1

    doi = published.get("doi") or published.get("pids", {}).get("doi", {}).get("identifier", "unknown")
    record_url = published.get("links", {}).get("self_html", "")
    print(f"Published DOI: {doi}")
    print(f"Concept DOI: 10.5281/zenodo.20780569")
    print(f"Record: {record_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
