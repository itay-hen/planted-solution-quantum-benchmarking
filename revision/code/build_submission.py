"""Regenerate the v21 comparison and build the clean, marked and response PDFs.

Requires TeX Live with REVTeX 4.2, pdflatex, bibtex and latexdiff on PATH.
The baseline source and its historical label numbers are bundled in diff/.
"""
from pathlib import Path
import re
import subprocess
import tempfile

REV = Path(__file__).resolve().parent.parent


def run(args):
    result = subprocess.run(args, cwd=REV, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed:\n{result.stdout[-5000:]}\n{result.stderr[-2000:]}")
    return result.stdout


def make_diff():
    old = (REV / "diff/qb_gs_v21.tex").read_text()
    new = (REV / "qb_gs_v22.tex").read_text()
    labels = set(re.findall(r"\\label\{([^}]+)\}", new))
    historical = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}",
                                 (REV / "diff/qb_gs_v21.aux").read_text()))
    # References in withdrawn text must identify the old numbering, rather than
    # pointing to nonexistent labels or silently borrowing the new numbering.
    def old_ref(match):
        command, label = match.groups()
        if label in labels:
            return match.group(0)
        if label not in historical:
            raise ValueError(f"missing historical label: {label}")
        number = historical[label]
        return f"({number}, v21)" if command == "eqref" else f"{number} (v21)"
    old = re.sub(r"\\(eqref|ref)\{([^}]+)\}", old_ref, old)
    for name in ("fig_construction_pipeline", "fig_degeneracy", "fig_intensive_gap"):
        old = old.replace("{" + name + ".pdf}", "{diff/" + name + ".pdf}")
    with tempfile.TemporaryDirectory(prefix="qb-gs-latexdiff-") as temp:
        baseline = Path(temp) / "v21.tex"
        baseline.write_text(old)
        diff = run(["latexdiff", "--graphics-markup=none", "--disable-citation-markup",
                    r"--config=PICTUREENV=(?:picture[\w\d*@]*|tikzpicture[\w\d*@]*|DIFnomarkup|table\*?)",
                    "--label=v21 (submitted)", "--label=v22 (corrected)",
                    str(baseline), str(REV / "qb_gs_v22.tex")])
    diff = diff.replace(r"\bibliography{refs_v22}", r"\bibliography{refs_v22,diff/old_removed}")
    (REV / "qb_gs_v22_diff.tex").write_text(diff)


def main():
    make_diff()
    for stem in ("qb_gs_v22", "qb_gs_v22_diff", "response_to_referees"):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex"])
        if stem != "response_to_referees":
            run(["bibtex", stem])
        for _ in range(2):
            run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex"])
        log = (REV / (stem + ".log")).read_text(errors="replace")
        if "undefined references" in log or "undefined citations" in log:
            raise RuntimeError(f"{stem}: unresolved references or citations; inspect the log")
        print(f"built {stem}.pdf", flush=True)


if __name__ == "__main__":
    main()
