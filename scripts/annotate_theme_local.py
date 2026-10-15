"""Open an existing blind theme sheet on the annotator's own computer.

This entry point uses only the Python standard library and the checked-out
repository. It never creates a new sample or reads the full corpus.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from horus_benchmark.audit.theme_annotation_ui import serve_annotation_ui  # noqa: E402


def load_taxonomy(path: Path) -> dict[str, tuple[str, ...]]:
    taxonomy: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            area, theme = row["domain"].strip(), row["theme"].strip()
            if not area or not theme:
                raise ValueError(f"Taxonomia incompleta em {path}")
            taxonomy.setdefault(area, set()).add(theme)
    if len(taxonomy) != 6:
        raise ValueError(f"Esperadas seis áreas na taxonomia: {path}")
    return {area: tuple(sorted(themes)) for area, themes in sorted(taxonomy.items())}


def resume_pilot() -> Path:
    root = ROOT / "results" / "theme_validation"
    candidates = list(root.glob("*_human-theme-validation-v1-pilot_*/pilot_annotation.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Nenhum piloto local encontrado. Informe --sheet com o CSV recebido."
        )
    ranked: list[tuple[int, float, Path]] = []
    id_sets: set[frozenset[str]] = set()
    for path in candidates:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream)
            if not {"record_id", "theme_human"}.issubset(rows.fieldnames or []):
                raise ValueError(f"CSV de piloto incompleto: {path}")
            ids: set[str] = set()
            completed = 0
            for row in rows:
                if not row["record_id"] or row["record_id"] in ids:
                    raise ValueError(f"ID vazio ou duplicado no piloto: {path}")
                ids.add(row["record_id"])
                completed += bool(row["theme_human"].strip())
            id_sets.add(frozenset(ids))
        ranked.append((completed, path.stat().st_mtime, path))
    if len(id_sets) != 1:
        raise ValueError(
            "Há pilotos com IDs diferentes. Use --sheet para escolher um arquivo específico."
        )
    completed, _, chosen = max(ranked)
    print(f"Retomando piloto: {chosen} ({completed} respostas salvas)", flush=True)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pilot", action="store_true", help="Retomar o piloto local com mais respostas")
    target.add_argument("--sheet", type=Path, help="CSV cego recebido para anotação")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    sheet = resume_pilot() if args.pilot else args.sheet.expanduser().resolve()
    taxonomy = load_taxonomy(ROOT / "themes_full_dataset.csv")
    serve_annotation_ui(
        SimpleNamespace(taxonomy=taxonomy), sheet,
        port=args.port, open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
