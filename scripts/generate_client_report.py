#!/usr/bin/env python3
"""Score photo folders and write a client PDF report.

Builds HTML in a temp directory, prints it to PDF with Chrome, then keeps
only the PDF.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import date
from io import BytesIO
from pathlib import Path

from PIL import Image

from attire_verification.cli import IMAGE_EXTENSIONS, run_verify
from attire_verification.models import VerifyResult
from attire_verification.roles import Role, parse_role, requires_id_badge

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHOTO_DIRS = [
    Path("/Users/afridee/Downloads/Photos"),
    Path("/Users/afridee/Downloads/BR Attire Verification_Sample Photos"),
]
DEFAULT_OUTPUT = ROOT / "reports" / "attire_verification_report.pdf"
DEFAULT_ROLE = Role.FC.value
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

REGION_META = {
    "upper": {
        "title": "Shirt / top",
        "pass": "Tucked-in solid-colour formal shirt (white, light blue, mint, beige, navy, or light gray), or the official black polo with purple sleeve trim.",
        "fail": "Anything other than a tucked-in solid-colour formal shirt or the official polo.",
    },
    "lower": {
        "title": "Trousers",
        "pass": "Black or navy trousers.",
        "fail": "Casual trousers (not black or navy).",
    },
    "feet": {
        "title": "Shoes",
        "pass": "Closed shoes, loafers, or plain (single-colour) sneakers.",
        "fail": "Sandals, slides, or bright/neon shoes.",
    },
    "chest": {
        "title": "ID badge",
        "pass": "Not required for this role — automatically awarded.",
        "fail": "Visible ID badge required only for BR / BR Supervisor.",
    },
}

FAIL_COPY = {
    "incomplete_body_in_frame": (
        "Photo is not a full-body shot",
        "Head or feet are not clearly visible, or the person is too small in the frame. "
        "This is a retake, not a dress-code fail. Attire was not scored.",
    ),
    "casual_shirt": (
        "Shirt does not meet dress code",
        "The shirt is casual or non-formal. Required: a tucked-in solid-colour "
        "formal shirt (white, light blue, mint, beige, navy, or light gray), "
        "or the official black polo with purple sleeve trim.",
    ),
    "bright_color": (
        "Bright or neon colour",
        "Bright or neon colours are not allowed on shoes.",
    ),
    "open_footwear": (
        "Open footwear",
        "Sandals or slides are not allowed. Required: closed shoes, loafers, or plain sneakers.",
    ),
    "wrong_trousers": (
        "Trousers do not meet dress code",
        "Casual trousers were detected. Required: black or navy trousers.",
    ),
    "missing_id_badge": (
        "ID badge not visible",
        "A visible ID badge is required for BR / BR Supervisor. This should not apply in this evaluation.",
    ),
    "low_confidence": (
        "Could not confirm the item",
        "The model was not confident enough about what was worn (lighting, angle, or occlusion). "
        "The related region is scored 0.",
    ),
}


def collect_images(folders: list[Path]) -> list[Path]:
    images: list[Path] = []
    for folder in folders:
        if not folder.is_dir():
            raise SystemExit(f"Missing photo folder: {folder}")
        found = sorted(
            p for p in folder.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTENSIONS
        )
        if not found:
            raise SystemExit(f"No images found under {folder}")
        images.extend(found)
    return images


def source_label(path: Path) -> str:
    parts = path.parts
    if "Photos" in parts and "BR Attire Verification_Sample Photos" not in parts:
        return "Additional photos"
    if "Right Attire" in parts:
        return "Sample set · Right Attire folder"
    if "Wrong Attire" in parts:
        return "Sample set · Wrong Attire folder"
    return str(path.parent)


def parse_points(score: str) -> int:
    return int(score.split("/", 1)[0])


def outcome(result: VerifyResult) -> str:
    if "incomplete_body_in_frame" in result.failReasons:
        return "retake"
    if result.score == "100/100":
        return "pass"
    return "fail"


def preview_jpeg(src: Path, max_side: int = 900) -> bytes:
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, "JPEG", quality=85, optimize=True)
        return buf.getvalue()


def data_uri_jpeg(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def region_detected(result: VerifyResult, name: str) -> str:
    if result.regions is None:
        return "Not scored"
    region = getattr(result.regions, name)
    if region is None:
        return "Not scored"
    pct_val = round(region.topScore * 100)
    return f"{region.topLabel} ({pct_val}%)"


def points_for(result: VerifyResult, name: str) -> int:
    if result.regionPoints is None:
        return 0
    return int(getattr(result.regionPoints, name))


def fail_items(result: VerifyResult) -> list[tuple[str, str, str]]:
    items = []
    for code in result.failReasons:
        title, detail = FAIL_COPY.get(
            code,
            (code.replace("_", " ").title(), "See technical notes."),
        )
        items.append((code, title, detail))
    return items


def pct(n: int, total: int) -> str:
    if not total:
        return "0%"
    return f"{round(100 * n / total)}%"


def render_html(rows: list[dict], role: str) -> str:
    need_badge = requires_id_badge(role)
    total = len(rows)
    n_pass = sum(1 for r in rows if r["outcome"] == "pass")
    n_fail = sum(1 for r in rows if r["outcome"] == "fail")
    n_retake = sum(1 for r in rows if r["outcome"] == "retake")
    scored = [r for r in rows if r["outcome"] != "retake"]
    avg = round(sum(r["points"] for r in scored) / len(scored), 1) if scored else 0

    reason_counter: Counter[str] = Counter()
    for r in rows:
        for code, title, _ in r["fails"]:
            if code != "incomplete_body_in_frame":
                reason_counter[title] += 1
    top_reasons = reason_counter.most_common(6)

    cards = []
    index_rows = []
    for i, r in enumerate(rows, start=1):
        badge_text = {
            "pass": "Pass · 100/100",
            "fail": f"Needs attention · {r['score']}",
            "retake": "Retake photo · 0/100",
        }[r["outcome"]]

        if r["fails"]:
            fail_html = "".join(
                f'<div class="fail-item"><strong>{html.escape(title)}</strong>'
                f"<p>{html.escape(detail)}</p></div>"
                for _, title, detail in r["fails"]
            )
        else:
            fail_html = '<p class="ok-note">No issues found. All dress-code checks passed.</p>'

        region_html = []
        for key, meta in REGION_META.items():
            pts = r["region_points"][key]
            passed = pts == 25
            if r["outcome"] == "retake":
                status = "Not scored"
                klass = "region retake"
            else:
                status = "Pass" if passed else "Fail"
                klass = "region pass" if passed else "region fail"
            if key == "chest" and r["outcome"] != "retake" and not need_badge:
                note = "Auto-pass (badge not required)"
            else:
                note = r["detected"][key]
            region_html.append(
                f"""
                <div class="{klass}">
                  <div class="region-head">
                    <span>{html.escape(meta["title"])}</span>
                    <span class="pts">{pts}/25</span>
                  </div>
                  <div class="region-status">{status}</div>
                  <div class="region-note">{html.escape(note)}</div>
                </div>
                """
            )

        polo = r["polo"]
        polo_html = ""
        if polo is not None:
            match_txt = "matches official polo" if polo["matched"] else "does not match official polo"
            polo_html = (
                f'<p class="polo">Official-polo similarity: <strong>{polo["score"]:.2f}</strong> '
                f'(threshold {polo["threshold"]:.2f}) — {match_txt}.</p>'
            )

        cards.append(
            f"""
            <article class="card" id="photo-{i}">
              <div class="card-head">
                <div>
                  <div class="photo-id">Photo {i:02d} of {total:02d}</div>
                  <h2>{html.escape(r["filename"])}</h2>
                  <div class="meta">{html.escape(r["source"])}</div>
                </div>
                <div class="badge {r["outcome"]}">{html.escape(badge_text)}</div>
              </div>
              <div class="card-body">
                <div class="photo-wrap">
                  <img src="{r["asset"]}" alt="{html.escape(r["filename"])}">
                </div>
                <div class="detail">
                  <div class="score-line">
                    <div class="score-num">{html.escape(r["score"])}</div>
                    <div class="score-bar"><span style="width:{r["points"]}%"></span></div>
                  </div>
                  <div class="regions">{"".join(region_html)}</div>
                  {polo_html}
                  <h3>Why this score</h3>
                  {fail_html}
                </div>
              </div>
            </article>
            """
        )

        index_rows.append(
            f"""
            <tr>
              <td><a href="#photo-{i}">{i:02d}</a></td>
              <td><a href="#photo-{i}">{html.escape(r["filename"])}</a></td>
              <td>{html.escape(r["source"])}</td>
              <td class="num">{html.escape(r["score"])}</td>
              <td><span class="pill {r["outcome"]}">{r["outcome"].title()}</span></td>
            </tr>
            """
        )

    reason_rows = "".join(
        f"<tr><td>{html.escape(title)}</td><td class='num'>{count}</td></tr>"
        for title, count in top_reasons
    ) or "<tr><td colspan='2'>No dress-code fails (only passes / retakes).</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Attire Verification Report</title>
  <style>
    :root {{
      --ink: #1c2430;
      --muted: #5b6777;
      --line: #d9e0ea;
      --bg: #f4f6f9;
      --card: #ffffff;
      --pass: #1b7f4e;
      --pass-bg: #e7f6ee;
      --fail: #b42318;
      --fail-bg: #fdecea;
      --retake: #9a6700;
      --retake-bg: #fff6dd;
      --brand: #12325c;
      --accent: #2b6cb0;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.45;
    }}
    .page {{ max-width: 1100px; margin: 0 auto; padding: 28px 20px 64px; }}
    header.hero {{
      background: var(--brand);
      color: #fff;
      border-radius: 16px;
      padding: 28px 32px;
      margin-bottom: 24px;
    }}
    header.hero h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: -0.02em; }}
    header.hero p {{ margin: 0; color: #d7e3f4; max-width: 70ch; }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin: 20px 0 24px;
    }}
    .kpi {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
    }}
    .kpi .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .kpi .value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
    .kpi.pass .value {{ color: var(--pass); }}
    .kpi.fail .value {{ color: var(--fail); }}
    .kpi.retake .value {{ color: var(--retake); }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 20px 22px;
      margin-bottom: 20px;
    }}
    h2, h3 {{ margin: 0 0 10px; }}
    h2 {{ font-size: 20px; color: var(--brand); }}
    h3 {{ font-size: 14px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }}
    .legend {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px 24px;
      font-size: 14px;
    }}
    .legend dt {{ font-weight: 700; }}
    .legend dd {{ margin: 0 0 8px; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }}
    td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    a {{ color: var(--accent); text-decoration: none; }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 600;
    }}
    .pill.pass, .badge.pass {{ background: var(--pass-bg); color: var(--pass); }}
    .pill.fail, .badge.fail {{ background: var(--fail-bg); color: var(--fail); }}
    .pill.retake, .badge.retake {{ background: var(--retake-bg); color: var(--retake); }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px;
      margin: 0 0 18px;
      break-inside: avoid;
      page-break-inside: avoid;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .photo-id {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
    .card-head h2 {{ margin: 2px 0 4px; font-size: 18px; color: var(--ink); }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    .badge {{
      border-radius: 999px;
      padding: 8px 12px;
      font-weight: 700;
      white-space: nowrap;
      font-size: 13px;
    }}
    .card-body {{
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 18px;
      align-items: start;
    }}
    .photo-wrap {{
      background: #0f1724;
      border-radius: 10px;
      overflow: hidden;
      min-height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .photo-wrap img {{
      max-width: 100%;
      max-height: 420px;
      display: block;
      object-fit: contain;
    }}
    .score-line {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }}
    .score-num {{ font-size: 28px; font-weight: 800; color: var(--brand); min-width: 92px; }}
    .score-bar {{
      flex: 1;
      height: 10px;
      background: #e8edf4;
      border-radius: 999px;
      overflow: hidden;
    }}
    .score-bar span {{
      display: block;
      height: 100%;
      background: var(--brand);
    }}
    .regions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .region {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      background: #fafbfd;
    }}
    .region.pass {{ border-color: #b7e2c9; background: var(--pass-bg); }}
    .region.fail {{ border-color: #f4c7c3; background: var(--fail-bg); }}
    .region.retake {{ border-color: #ead79a; background: var(--retake-bg); }}
    .region-head {{ display: flex; justify-content: space-between; font-weight: 700; font-size: 13px; }}
    .region-status {{ font-size: 12px; font-weight: 700; margin: 2px 0; }}
    .region.pass .region-status {{ color: var(--pass); }}
    .region.fail .region-status {{ color: var(--fail); }}
    .region.retake .region-status {{ color: var(--retake); }}
    .region-note, .polo {{ font-size: 12px; color: var(--muted); }}
    .fail-item {{
      background: var(--fail-bg);
      border-radius: 8px;
      padding: 8px 10px;
      margin: 0 0 8px;
    }}
    .fail-item p, .ok-note {{ margin: 4px 0 0; font-size: 13px; color: var(--muted); }}
    .ok-note {{
      background: var(--pass-bg);
      color: var(--pass);
      padding: 8px 10px;
      border-radius: 8px;
    }}
    .note {{ color: var(--muted); font-size: 13px; }}
    footer {{ color: var(--muted); font-size: 12px; margin-top: 12px; }}
    @media print {{
      body {{ background: #fff; }}
      .page {{ max-width: none; padding: 0; }}
      header.hero {{ border-radius: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
      .card, .panel, .kpi, .region, .badge, .pill, .fail-item, .ok-note {{
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }}
      a {{ color: inherit; }}
      .card {{ page-break-before: always; }}
      @page {{ size: A4; margin: 12mm; }}
    }}
    @media (max-width: 800px) {{
      .kpis {{ grid-template-columns: 1fr 1fr; }}
      .card-body, .legend {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <h1>Attire Verification Report</h1>
      <p>
        Offline evaluation of {total} staff photos on {date.today():%d %B %Y}.
        Role used: <strong>{html.escape(role)}</strong>.
        {"A visible ID badge is required." if need_badge else "A visible ID badge is not required, so that check is automatically awarded 25 points."}
        Folder names such as “Right Attire” / “Wrong Attire” were not used as expected answers.
      </p>
    </header>

    <section class="kpis">
      <div class="kpi"><div class="label">Photos reviewed</div><div class="value">{total}</div></div>
      <div class="kpi pass"><div class="label">Passed (100/100)</div><div class="value">{n_pass}</div></div>
      <div class="kpi fail"><div class="label">Needs attention</div><div class="value">{n_fail}</div></div>
      <div class="kpi retake"><div class="label">Retake photo</div><div class="value">{n_retake}</div></div>
    </section>

    <section class="panel">
      <h2>How to read the score</h2>
      <p>Each photo can earn <strong>100 points</strong>: 25 for shirt, 25 for trousers, 25 for shoes, and 25 for ID badge. A check either passes (25) or fails (0) — there is no partial credit.</p>
      <dl class="legend">
        <div><dt>Shirt / top (25)</dt><dd>{html.escape(REGION_META["upper"]["pass"])} Fails if: {html.escape(REGION_META["upper"]["fail"])}</dd></div>
        <div><dt>Trousers (25)</dt><dd>{html.escape(REGION_META["lower"]["pass"])} Fails if: {html.escape(REGION_META["lower"]["fail"])}</dd></div>
        <div><dt>Shoes (25)</dt><dd>{html.escape(REGION_META["feet"]["pass"])} Fails if: {html.escape(REGION_META["feet"]["fail"])}</dd></div>
        <div><dt>ID badge (25)</dt><dd>{"Visible ID badge required for this role." if need_badge else html.escape(REGION_META["chest"]["pass"])}</dd></div>
      </dl>
      <p class="note">Average score among photos that could be scored (excluding retakes): <strong>{avg}/100</strong>. Pass rate: <strong>{pct(n_pass, total)}</strong> of all photos.</p>
    </section>

    <section class="panel">
      <h2>Most common issues</h2>
      <table>
        <thead><tr><th>Issue</th><th class="num">Photos</th></tr></thead>
        <tbody>{reason_rows}</tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Photo index</h2>
      <table>
        <thead>
          <tr>
            <th>#</th><th>File</th><th>Source</th><th class="num">Score</th><th>Result</th>
          </tr>
        </thead>
        <tbody>
          {"".join(index_rows)}
        </tbody>
      </table>
    </section>

    <h2 style="margin: 8px 4px 12px;">Photo-by-photo results</h2>
    {"".join(cards)}

    <footer>
      Generated by the attire verification eval (MediaPipe pose framing + FashionSigLIP clothing scores).
      Role used: {html.escape(role)}. ID-badge rule: {"on" if need_badge else "off"}.
    </footer>
  </div>
</body>
</html>
"""


def chrome_pdf(html_path: Path, pdf_path: Path) -> None:
    if not CHROME.exists():
        raise SystemExit("Google Chrome not found; cannot convert HTML to PDF.")
    cmd = [
        str(CHROME),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--no-first-run",
        "--disable-extensions",
        f"--print-to-pdf={pdf_path}",
        "--virtual-time-budget=180000",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score photo folders and write a client PDF report."
    )
    parser.add_argument(
        "--dir",
        dest="dirs",
        type=Path,
        action="append",
        help="Photo folder (repeatable). Defaults to the two local sample folders.",
    )
    parser.add_argument(
        "--role",
        default=DEFAULT_ROLE,
        help=f"Staff role for every photo (default: {DEFAULT_ROLE}; ID badge not required).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PDF output path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    role = parse_role(args.role).value
    folders = args.dirs or list(DEFAULT_PHOTO_DIRS)
    images = collect_images(folders)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    print(f"Scoring {len(images)} photos as role={role}", flush=True)
    for i, image in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] {image}", flush=True)
        result = run_verify(image, role=role)
        polo = result.poloMatch.model_dump() if result.poloMatch is not None else None
        rows.append(
            {
                "filename": image.name,
                "source": source_label(image),
                "asset": data_uri_jpeg(preview_jpeg(image)),
                "score": result.score,
                "points": parse_points(result.score),
                "outcome": outcome(result),
                "fails": fail_items(result),
                "region_points": {
                    "upper": points_for(result, "upper"),
                    "lower": points_for(result, "lower"),
                    "feet": points_for(result, "feet"),
                    "chest": points_for(result, "chest"),
                },
                "detected": {
                    "upper": region_detected(result, "upper"),
                    "lower": region_detected(result, "lower"),
                    "feet": region_detected(result, "feet"),
                    "chest": region_detected(result, "chest"),
                },
                "polo": polo,
            }
        )

    with tempfile.TemporaryDirectory(prefix="attire_report_") as tmp:
        tmp_dir = Path(tmp)
        html_path = tmp_dir / "attire_verification_report.html"
        pdf_tmp = tmp_dir / "attire_verification_report.pdf"
        html_path.write_text(render_html(rows, role), encoding="utf-8")
        print(f"Wrote HTML {html_path} ({html_path.stat().st_size} bytes)", flush=True)
        print("Converting HTML to PDF via Chrome…", flush=True)
        chrome_pdf(html_path, pdf_tmp)
        shutil.copy2(pdf_tmp, output)
        print(f"Wrote {output}", flush=True)

    summary = {
        "total": len(rows),
        "pass": sum(1 for r in rows if r["outcome"] == "pass"),
        "fail": sum(1 for r in rows if r["outcome"] == "fail"),
        "retake": sum(1 for r in rows if r["outcome"] == "retake"),
        "pdf": str(output),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
