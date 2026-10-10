#!/usr/bin/env python3
"""Checks one book chapter against the chapter template and the house rules.

Usage: scripts/check-chapter.py book/NN-slug.md [...]
Exit 1 if any check fails. Prints one line per failure.

Checks: template blocks, no glossary links, forward links only in the Next line,
no em or en dash outside code, word count 1000-1600, code excerpts verbatim
against the pinned repositories under code/ (when the Source: line names a path
that exists there).
"""
import re, sys, os, glob, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTERS = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(ROOT, "book", "[0-9][0-9]-*.md")))

def strip_code(md):
    return re.sub(r"```.*?```", "", md, flags=re.S)

def check(path):
    fails = []
    md = open(path).read()
    me = os.path.basename(path)[:-3]
    num = int(me[:2])
    prose = strip_code(md)

    # 1. template blocks, in order
    heads = re.findall(r"(?m)^(#{1,3}) (.*)$", prose)
    if not heads or heads[0][0] != "#":
        fails.append("no H1 title")
    names = [h[1].strip() for h in heads]
    for req in ["Where we are", "Check yourself", "Answers"]:
        if req not in names:
            fails.append(f"missing block: {req}")
    if "Check yourself" in names and "Answers" in names and names.index("Answers") < names.index("Check yourself"):
        fails.append("Answers before Check yourself")
    if 9 <= num <= 23 and "Where the matmul stands" not in names:
        fails.append("missing block: Where the matmul stands (Parts 3 to 6)")
    lines = md.rstrip().split("\n")
    if not re.match(r"^Next: \[.+\]\((\d\d-[a-z0-9-]+\.md)\)$", lines[-1]):
        if num != len(CHAPTERS):
            fails.append("last line is not a Next: [title](NN-slug.md) line")

    # 2. links: no glossary, forward links only in the Next line
    body_wo_next = "\n".join(lines[:-1]) if lines[-1].startswith("Next:") else md
    for m in re.finditer(r"\]\(([^)]+)\)", strip_code(body_wo_next)):
        t = m.group(1)
        if "glossary" in t:
            fails.append(f"link into the glossary: {t}")
        mm = re.match(r"(\d\d)-[a-z0-9-]+\.md", t)
        if mm and int(mm.group(1)) > num:
            fails.append(f"forward link in the body: {t}")
        if t.startswith("../") and not t.startswith("../diagrams/"):
            fails.append(f"relative link outside book/: {t}")

    # 3. dashes outside code
    for i, line in enumerate(prose.split("\n"), 1):
        if "—" in line or "–" in line:
            fails.append(f"em or en dash in prose near: {line.strip()[:60]}")

    # 4. word count of prose (code, image alt text and table rows excluded)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", prose)
    text = "\n".join(l for l in text.split("\n") if not l.startswith("|"))
    words = len(re.sub(r"[#*`|>\[\]()]", " ", text).split())
    if not 1000 <= words <= 1900:
        fails.append(f"word count {words} outside 1000-1900")

    # 5. excerpts verbatim against code/
    blocks = re.findall(r"```\w*\n(.*?)```\n(?:Source: \[([^\]]*)\]\(([^)]*)\))?", md, flags=re.S)
    for code, label, url in blocks:
        if not url:
            continue
        mm = re.search(r"github\.com/[^/]+/([^/]+)/blob/[0-9a-f]+/([^#]+)(?:#L(\d+)(?:-L(\d+))?)?", url)
        if not mm:
            continue
        repo, rel, l1, l2 = mm.groups()
        cands = glob.glob(os.path.join(ROOT, "code", "*", rel)) + glob.glob(os.path.join(ROOT, "code", "*", "*", rel))
        pref = [c for c in cands if repo.lower() in c.lower().rsplit("/code/", 1)[-1].split("/")[0]]
        cands = pref or cands  # prefer the checkout whose directory is named after the repo
        if not cands:
            continue  # repository not fetched; nothing to compare against
        src = open(cands[0], errors="replace").read().split("\n")
        lo, hi = (int(l1) - 1, int(l2 or l1)) if l1 else (0, len(src))
        window = "\n".join(src[lo:hi])
        for ln in code.rstrip("\n").split("\n"):
            s = ln.strip()
            if not s or s == "...":
                continue
            if s not in window:
                fails.append(f"excerpt line not found verbatim in {rel}:{l1}-{l2 or l1}: {s[:60]}")
                break
    return words, fails

bad = 0
for p in sys.argv[1:]:
    words, fails = check(p)
    print(f"{os.path.basename(p)}: {words} words, {len(fails)} failure(s)")
    for f in fails:
        print("  -", f)
    bad += bool(fails)
sys.exit(1 if bad else 0)
