"""Create a blank second-annotator sheet from the fixed 120-text pilot.

The source sheet is never changed. Existing output is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from annotate_theme_local import resume_pilot


BLIND_COLUMNS = ["record_id", "broad_area", "content", "theme_human", "annotation_note"]
REQUIRED = set(BLIND_COLUMNS)


def prepare(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        raise ValueError("A segunda folha precisa ter outro caminho")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if not REQUIRED.issubset(columns):
            raise ValueError(f"Colunas ausentes no piloto: {sorted(REQUIRED - set(columns))}")
        rows = list(reader)
    if len(rows) != 120:
        raise ValueError(f"Esperados 120 textos no piloto, encontrados {len(rows)}")
    ids = [row["record_id"] for row in rows]
    areas = Counter(row["broad_area"] for row in rows)
    if len(set(ids)) != 120 or not all(ids) or len(areas) != 6 or set(areas.values()) != {20}:
        raise ValueError("O piloto não contém 120 IDs únicos, 20 por área")
    blind_rows = [
        {
            "record_id": row["record_id"],
            "broad_area": row["broad_area"],
            "content": row["content"],
            "theme_human": "",
            "annotation_note": "",
        }
        for row in rows
    ]
    with destination.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=BLIND_COLUMNS)
        writer.writeheader()
        writer.writerows(blind_rows)
    print(f"Segunda folha: {destination}")
    print("120 textos; 20 por área; respostas do primeiro anotador removidas")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true", help="Usar o piloto local com mais respostas")
    group.add_argument("--source", type=Path, help="CSV da primeira anotação")
    parser.add_argument("--output", type=Path, help="Destino; padrão: second_annotation.csv ao lado do piloto")
    args = parser.parse_args()
    source = resume_pilot() if args.pilot else args.source
    destination = args.output or source.parent / "second_annotation.csv"
    prepare(source, destination)


if __name__ == "__main__":
    main()
