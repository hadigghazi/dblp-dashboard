"""
Builds what the site's research page reads: paper.json (the page's content) and a BibTeX entry, next to
the compiled PDF.

The title, author and abstract are taken from main.tex itself, and the page count from the LaTeX log, so
the page always describes the PDF it offers; site.json, kept beside the paper, holds what the page says
around it (findings, results, the figure's points). Run by the Paper workflow after the PDF is built:

    python publish.py <paper dir> <out dir>      (env: PAPER_COMMIT, PAPER_DATE)
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

PDF_NAME = "dblpqa-paper.pdf"
BIB_NAME = "dblpqa-paper.bib"
SITE_URL = "https://dblp.hadighazi.com/downloads/research/dblpqa/"
MACROS = {r"\dq{}": "DBLP-QA", r"\fresh{}": "DBLP-QA-Fresh"}


def plain(tex):
    """LaTeX source text to plain text: the few constructs the title and abstract use."""
    for macro, text in MACROS.items():
        tex = tex.replace(macro, text)
    tex = re.sub(r"(?<!\\)%.*", "", tex)                      # comments - but not \%, a percent sign
    tex = re.sub(r"~?\\cite\{[^}]*\}", "", tex)
    tex = re.sub(r"\\(?:textbf|textit|emph|texttt)\{([^{}]*)\}", r"\1", tex)
    tex = tex.replace(r"\%", "%").replace(r"\&", "&").replace("---", "—").replace("--", "–")
    tex = tex.replace("~", " ").replace(r"\,", " ").replace(r"\ldots{}", "…").replace(r"\ldots", "…")
    tex = re.sub(r"\$([^$]*)\$", r"\1", tex)                   # inline maths: $+0.30$ -> +0.30
    tex = tex.replace("``", "“").replace("''", "”")
    return " ".join(tex.split())


def extract(source):
    title = re.search(r"\\title\{(.*?)\}\s*\\author", source, re.S)
    name = re.search(r"\\IEEEauthorblockN\{([^}]*)\}", source)
    affiliation = re.search(r"\\IEEEauthorblockA\{\\textit\{([^}]*)\}", source)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.S)
    keywords = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", source, re.S)
    if not (title and name and abstract):
        raise SystemExit("main.tex: could not find the title, author or abstract")
    return {"title": plain(title.group(1)), "author": plain(name.group(1)),
            "affiliation": plain(affiliation.group(1)) if affiliation else None,
            "abstract": plain(abstract.group(1)),
            "keywords": [k.strip() for k in plain(keywords.group(1)).split(",")] if keywords else []}


def pages(log_path):
    found = re.search(r"Output written on \S+ \((\d+) pages?", Path(log_path).read_text(errors="replace")) \
        if Path(log_path).exists() else None
    return int(found.group(1)) if found else None


def bibtex(meta, year):
    surname = meta["author"].split()[-1].lower()
    return (f"@misc{{{surname}{year}dblpqa,\n"
            f"  author       = {{{meta['author']}}},\n"
            f"  title        = {{{meta['title']}}},\n"
            f"  year         = {{{year}}},\n"
            f"  note         = {{Manuscript, {meta['affiliation'] or ''}}},\n"
            f"  howpublished = {{\\url{{{SITE_URL}{PDF_NAME}}}}}\n"
            f"}}\n")


def main(paper_dir, out_dir):
    paper_dir, out_dir = Path(paper_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = extract((paper_dir / "main.tex").read_text(encoding="utf-8"))
    site = json.loads((paper_dir / "site.json").read_text(encoding="utf-8"))
    updated = os.environ.get("PAPER_DATE") or ""
    year = updated[:4] or "2026"
    shutil.copyfile(paper_dir / "main.pdf", out_dir / PDF_NAME)
    (out_dir / BIB_NAME).write_text(bibtex(meta, year), encoding="utf-8")
    payload = dict(site, paper=dict(meta, pdf=PDF_NAME, bib=BIB_NAME, pages=pages(paper_dir / "main.log"),
                                    bytes=(out_dir / PDF_NAME).stat().st_size, updated=updated,
                                    commit=(os.environ.get("PAPER_COMMIT") or "")[:7]))
    (out_dir / "paper.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{meta['title']} - {payload['paper']['pages']} pages, {payload['paper']['bytes']} bytes")


if __name__ == "__main__":
    main(*sys.argv[1:3])
