"""Prepare a content-addressed software supplement and package submission files.

Run --prepare after tests/data regeneration and before building the PDFs. Run
--archive after the final builds. Neither command publishes anything remotely.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

REV = Path(__file__).resolve().parent.parent
VERSION = "v22-r1"


def manifest():
    paths = [REV / "requirements.txt"]
    for folder in ("code", "vendor", "data"):
        paths.extend(p for p in (REV / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts
                     and p.suffix not in (".pyc", ".DS_Store"))
    hashes = {p.relative_to(REV).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(paths)}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"version": VERSION, "source_data_sha256": fingerprint, "files": hashes}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--archive", action="store_true")
    args = ap.parse_args()
    current = manifest()
    target = REV / "MANIFEST.json"
    if args.prepare:
        target.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        bib = REV / "refs_v22.bib"
        text = bib.read_text()
        start = text.index("@misc{Hen2026PlantedSolutionBenchmarking,")
        end = text.index("\n}", start)
        entry = text[start:end]
        note = (r"  note      = {Revision software/data snapshot \texttt{" + VERSION
                + r"}, SHA-256 prefix \texttt{" + current["source_data_sha256"][:16]
                + r"}; supplied as Supplemental Material. Repository hosts the earlier implementation}")
        entry = re.sub(r"  note\s*=.*", lambda _: note + ",", entry)
        bib.write_text(text[:start] + entry + text[end:])
        print("prepared", VERSION, current["source_data_sha256"])
        return
    if not target.exists() or json.loads(target.read_text()) != current:
        ap.error("software/data changed: rerun --prepare and rebuild the manuscript before archiving")
    out = REV / "release"
    out.mkdir(exist_ok=True)
    archive = out / "qb-gs-v22-r1.zip"
    allowed_root = {".md", ".tex", ".bib", ".pdf", ".json", ".txt", ".bbl"}
    files = [p for p in REV.iterdir() if p.is_file() and p.suffix in allowed_root]
    for folder in ("code", "vendor", "data", "figs", "diff"):
        files.extend(p for p in (REV / folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.name != ".DS_Store"
                     and p.suffix not in (".pyc", ".log", ".out", ".blg"))
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            info = zipfile.ZipInfo("qb-gs-v22-r1/" + p.relative_to(REV).as_posix(),
                                   date_time=(2026, 11, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, p.read_bytes())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / "SHA256SUMS").write_text(digest + "  " + archive.name + "\n")
    print("packaged", archive.name, digest)


if __name__ == "__main__":
    main()
