"""Local blind annotation form for pilot and main theme review CSV files."""

from __future__ import annotations

import csv
import hashlib
import html
import os
import secrets
import tempfile
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from horus_benchmark.audit.theme_common import ThemeContext


REQUIRED_COLUMNS = ("record_id", "broad_area", "content", "theme_human", "annotation_note")
SPECIAL_OPTIONS = ("incerto", "fora_taxonomia")
AREA_NAMES = {
    "academic": "Acadêmica", "informational": "Informacional",
    "journalistic": "Jornalística", "juridical": "Jurídica",
    "literary": "Literária", "social": "Social",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AnnotationSheet:
    """Validated editable copy of one blind annotation CSV."""

    def __init__(self, path: Path, taxonomy: dict[str, tuple[str, ...]]) -> None:
        self.path = path.expanduser().resolve()
        self.taxonomy = taxonomy
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with self.path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            self.columns = reader.fieldnames or []
            if not all(col in self.columns for col in REQUIRED_COLUMNS):
                raise ValueError(f"CSV de anotação sem colunas obrigatórias: {self.path}")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError("CSV de anotação vazio")
        ids: set[str] = set()
        for row in self.rows:
            record_id = row["record_id"]
            area = row["broad_area"]
            label = row["theme_human"]
            if (None in row or any(row[col] is None for col in REQUIRED_COLUMNS)
                    or not record_id or record_id in ids or area not in taxonomy
                    or not row["content"] or not row["content"].strip()):
                raise ValueError(f"Linha inválida ou ID duplicado no CSV: {record_id}")
            if label and label not in (*taxonomy[area], *SPECIAL_OPTIONS):
                raise ValueError(f"Tema inválido no CSV: {record_id} / {label}")
            ids.add(record_id)
        self.fingerprint = _digest(self.path)

    def first_pending(self) -> int:
        return next((i for i, row in enumerate(self.rows) if not row["theme_human"]), 0)

    def completed(self) -> int:
        return sum(bool(row["theme_human"]) for row in self.rows)

    def save(self, index: int, record_id: str, theme: str, note: str) -> None:
        if not 0 <= index < len(self.rows) or self.rows[index]["record_id"] != record_id:
            raise ValueError("O registro mudou de posição; atualize a página")
        area = self.rows[index]["broad_area"]
        if theme not in (*self.taxonomy[area], *SPECIAL_OPTIONS):
            raise ValueError("Selecione um tema válido para a área")
        if len(note) > 4000:
            raise ValueError("A observação deve ter no máximo 4000 caracteres")
        if _digest(self.path) != self.fingerprint:
            raise RuntimeError("O CSV foi alterado fora desta interface; reinicie para evitar sobrescrita")
        old_theme, old_note = self.rows[index]["theme_human"], self.rows[index]["annotation_note"]
        self.rows[index]["theme_human"] = theme
        self.rows[index]["annotation_note"] = note
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8-sig", newline="", suffix=".csv",
                prefix=".annotation-", dir=self.path.parent, delete=False,
            ) as stream:
                temp_path = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=self.columns)
                writer.writeheader()
                writer.writerows(self.rows)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self.path)
            self.fingerprint = _digest(self.path)
        except Exception:
            self.rows[index]["theme_human"] = old_theme
            self.rows[index]["annotation_note"] = old_note
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise


def _page(sheet: AnnotationSheet, index: int, token: str) -> str:
    row = sheet.rows[index]
    selected = row["theme_human"]
    area = row["broad_area"]
    options = []
    for theme in (*sheet.taxonomy[area], *SPECIAL_OPTIONS):
        checked = " checked" if theme == selected else ""
        label = theme.replace("_", " ") if theme in SPECIAL_OPTIONS else theme
        options.append(
            f'<label class="option"><input type="radio" name="theme" '
            f'value="{html.escape(theme, quote=True)}"{checked} required>'
            f'<span>{html.escape(label)}</span></label>'
        )
    total = len(sheet.rows)
    completed = sheet.completed()
    previous = f'<a href="/?i={index - 1}">← Anterior</a>' if index else ""
    following = f'<a href="/?i={index + 1}">Pular →</a>' if index + 1 < total else ""
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anotação de temas — Hórus</title>
<style>
body{{font-family:system-ui, sans-serif;background:#f5f6fa;color:#202734;margin:0}}
main{{max-width:980px;margin:2rem auto;padding:0 1rem}}
.card{{background:white;border-radius:14px;padding:1.5rem;box-shadow:0 2px 12px #17203518;margin-bottom:1rem}}
h1{{margin:.1rem 0 .6rem;font-size:1.6rem}}.meta{{color:#526071;font-size:.95rem}}
.text{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:48vh;overflow:auto;line-height:1.6}}
.choices{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:.5rem}}
.option{{display:flex;align-items:flex-start;gap:.5rem;border:1px solid #d9dfe8;border-radius:8px;padding:.65rem;cursor:pointer}}
.option:has(input:checked){{border-color:#1c6ea4;background:#eaf5fc}}
.option input{{margin-top:.2rem}}textarea{{width:100%;min-height:3.5rem;box-sizing:border-box;margin-top:.5rem}}
.actions{{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-top:1rem}}
button{{background:#135f91;color:white;border:0;border-radius:8px;padding:.8rem 1.1rem;cursor:pointer;font-size:1rem}}
a{{color:#135f91;text-decoration:none}}progress{{width:100%;height:1rem}}
</style></head><body><main>
<div class="card"><h1>Anotação de temas</h1>
<div class="meta">{completed} de {total} concluídos · Texto {index + 1} de {total} · Área: {html.escape(AREA_NAMES[area])}</div>
<progress value="{completed}" max="{total}"></progress>
<div class="meta">ID: {html.escape(row['record_id'])}</div></div>
<div class="card text">{html.escape(row['content'])}</div>
<form class="card" method="post" action="/save">
<input type="hidden" name="token" value="{html.escape(token, quote=True)}">
<input type="hidden" name="index" value="{index}">
<input type="hidden" name="record_id" value="{html.escape(row['record_id'], quote=True)}">
<h2>Selecione o tema predominante</h2>
<div class="choices">{''.join(options)}</div>
<label for="note">Observação opcional</label>
<textarea id="note" name="note" maxlength="4000">{html.escape(row['annotation_note'])}</textarea>
<div class="actions">{previous}<button type="submit">Salvar e próximo →</button>{following}</div>
</form></main></body></html>"""


def serve_annotation_ui(
    context: ThemeContext, sheet_path: Path, *, port: int = 8765,
    open_browser: bool = True,
) -> Path:
    """Serve only on loopback; each answer is saved atomically to the sheet."""
    sheet = AnnotationSheet(sheet_path, context.taxonomy)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path != "/":
                self._send(404, "Página não encontrada")
                return
            try:
                index = int(parse_qs(parsed.query).get("i", [str(sheet.first_pending())])[0])
                if not 0 <= index < len(sheet.rows):
                    raise ValueError("Índice fora do CSV")
            except ValueError as exc:
                self._send(400, html.escape(str(exc)))
                return
            self._send(200, _page(sheet, index, token))

        def do_POST(self) -> None:
            if self.path != "/save":
                self._send(404, "Página não encontrada")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 32_000:
                    raise ValueError("Formulário inválido ou muito grande")
                fields = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
                if fields.get("token", [None])[0] != token:
                    raise ValueError("Sessão inválida")
                index = int(fields.get("index", [""])[0])
                sheet.save(index, fields.get("record_id", [""])[0],
                           fields.get("theme", [""])[0], fields.get("note", [""])[0])
            except (ValueError, RuntimeError, UnicodeDecodeError) as exc:
                self._send(400, html.escape(str(exc)))
                return
            if sheet.completed() == len(sheet.rows):
                pending = f"<p>Concluído: {len(sheet.rows)} textos. CSV salvo em {html.escape(str(sheet.path))}.</p>"
                self._send(200, "<!doctype html><html lang='pt-BR'><meta charset='utf-8'><title>Concluído</title>" + pending + "</html>")
                return
            next_index = next(
                (i for i in range(index + 1, len(sheet.rows)) if not sheet.rows[i]["theme_human"]),
                sheet.first_pending(),
            )
            self.send_response(303)
            self.send_header("Location", f"/?i={next_index}")
            self.end_headers()

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Anotação local: {url}\nCSV: {sheet.path}\nEncerre com Ctrl+C.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return sheet.path
