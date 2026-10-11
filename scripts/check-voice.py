#!/usr/bin/env python3
"""Checks that a book chapter reads like a book, not like a reference entry.

Usage: scripts/check-voice.py [--report] book/NN-slug.md [...] [src/manifest.mjs]
Exit 1 if any A (zero-tolerance) or B (threshold) check fails. --report prints a
per-chapter table of the measured signals instead.

A checks, zero tolerance: dashes; AI vocabulary; participle tails; negative
parallelism; copula avoidance; filler and signposting; hedge stacks; curly
quotes; title-case headings; generic send-off in the last paragraph.
B checks, thresholds: second person density; third-person stand-ins; bold-label
bullets; bullet share; boldface count; sentence-length spread and runs;
paragraph length and clipped runs; "Where we are" addresses you and names the
previous chapter; one closing sentence that points forward.
D check, manifest only: no blurb starts with an imperative verb.
"""
import re, sys, os, statistics

AI = r"\b(delve|delves|crucial|robust|leverage|leverages|landscape|tapestry|underscore|underscores|showcase|showcases|pivotal|foster|fosters|vibrant|testament|seamless|seamlessly|comprehensive|nuanced|intricate|intricacies|essentially|importantly|notably|interestingly|game-changer|cutting-edge|state-of-the-art|unlock|unlocks|empower|empowers)\b"
ING = r", (?:highlighting|underscoring|emphasizing|ensuring|reflecting|showcasing|enabling|allowing|contributing|fostering|demonstrating|illustrating)\b"
NEG = r"\bnot (?:only|just|merely|simply)\b[^.]{0,80}\bbut\b|\bit is not (?:a |an |about |that )[^.]{0,60}, it is\b|\bisn't (?:just|only)\b"
COP = r"\b(?:serves as|stands as|represents a|represents an|boasts|boasting)\b"
FILL = r"\b(?:In this chapter|This chapter (?:covers|will|explores|introduces)|Let's|let's|It is worth noting|It's worth noting|Note that|as mentioned above|as discussed above|as shown above|as we (?:saw|discussed|will see)|In order to|in order to|It should be noted|Needless to say|At the end of the day|In today's)\b"
HEDGE = r"\b(?:may potentially|could possibly|might possibly|it is important to (?:note|remember|understand)|it's important to)\b"
SENDOFF = r"\b(?:journey|exciting|bright future|the future (?:is|looks)|we hope|happy (?:coding|hacking)|good luck)\b"
THIRD = r"\b(?:the reader|readers|a CUDA engineer|a CUDA person|an engineer|a kernel engineer|the engineer|one must|one should|one can)\b"
IMPERATIVE = r"^(?:Name|Explain|Compute|Reason|Recognize|Read|Encode|Specialize|Time|Measure|Size|Hold|Say|Fold|Trace|Map|Tell|Convert|Ask|Disassemble|Place|Describe|Identify|List|Define|Show|Write|Build|Use|Learn|Understand)\b"

def prose_of(md):
    s = re.sub(r"```.*?```", "", md, flags=re.S)
    s = re.sub(r"(?m)^Source:.*$", "", s)
    s = re.sub(r"(?m)^!\[.*$", "", s)
    s = re.sub(r"`[^`\n]*`", "`c`", s)
    return s

def sentences(text):
    t = re.sub(r"\s+", " ", text)
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z`\"(])", t) if len(x.split()) > 2]

def paragraphs(text):
    out, cur = [], []
    for l in text.split("\n"):
        if l.strip() == "":
            if cur: out.append(" ".join(cur)); cur = []
        elif re.match(r"^#|^\s*([-*]|\d+\.)\s|^\||^Next:", l):
            if cur: out.append(" ".join(cur)); cur = []
        else:
            cur.append(l.strip())
    if cur: out.append(" ".join(cur))
    return out

def section(md, name):
    m = re.search(r"(?ms)^## " + re.escape(name) + r"\s*$(.*?)(?=^## |\Z)", md)
    return m.group(1) if m else ""

def measure(md):
    p = prose_of(md)
    lines = [l for l in p.split("\n") if l.strip()]
    bul = [l for l in lines if re.match(r"\s*([-*]|\d+\.)\s", l)]
    lbl = [l for l in bul if re.match(r"\s*([-*]|\d+\.)\s+\*\*", l)]
    body = re.sub(r"(?m)^#.*$", "", p)
    words = len(re.findall(r"\w+", body))
    you = len(re.findall(r"\byou\b|\byour\b|\byourself\b", body, flags=re.I))
    bold = len(re.findall(r"\*\*[^*\n]+\*\*", body))
    sl = [len(x.split()) for x in sentences(body)]
    sdev = statistics.pstdev(sl) if len(sl) > 1 else 0.0
    run = best = 1
    for a, b in zip(sl, sl[1:]):
        run = run + 1 if abs(a - b) <= 3 else 1
        best = max(best, run)
    paras = paragraphs(body)
    longest = max((len(x.split()) for x in paras), default=0)
    clip = cbest = 0
    for x in paras:
        clip = clip + 1 if len(sentences(x)) <= 1 else 0
        cbest = max(cbest, clip)
    return dict(words=words, youk=you / max(words, 1) * 1000, bold=bold, lbl=len(lbl),
                bulpct=len(bul) / max(len(lines), 1) * 100, sdev=sdev, run=best, longest=longest, clip=cbest)

def check_chapter(path):
    md = open(path).read()
    me = os.path.basename(path)[:-3]
    num = int(me[:2])
    p = prose_of(md)
    fails = []
    def hits(tag, rx, text=p, flags=0):
        for m in re.finditer(rx, text, flags):
            ln = text[:m.start()].count("\n") + 1
            fails.append(f"{tag}: line {ln}: '{m.group(0)}'")
    hits("A dash", r"—|–| -- ")
    hits("A ai-word", AI, flags=re.I)
    hits("A participle-tail", ING)
    hits("A negative-parallelism", NEG, flags=re.I)
    hits("A copula-avoidance", COP)
    hits("A filler", FILL)
    hits("A hedge", HEDGE, flags=re.I)
    hits("A curly-quote", r"[“”‘’]")
    for m in re.finditer(r"(?m)^#{2,3} (.+)$", p):
        ws = [w for w in re.findall(r"[A-Za-z]{4,}", m.group(1))]
        if len(ws) >= 3 and all(w[0].isupper() for w in ws):
            fails.append(f"A title-case heading: '{m.group(1)}'")
    paras = paragraphs(re.sub(r"(?m)^#.*$", "", p))
    if paras and re.search(SENDOFF, paras[-1], re.I):
        fails.append("A generic send-off in the last paragraph")

    m = measure(md)
    if m["youk"] < 3.0:
        fails.append(f"B second-person density {m['youk']:.1f}/1000 words (need 3.0)")
    hits("B third-person stand-in", THIRD)
    if m["lbl"] > 2:
        fails.append(f"B {m['lbl']} bold-label bullets (max 2)")
    if m["bulpct"] > 10:
        fails.append(f"B bullets are {m['bulpct']:.0f}% of body lines (max 10)")
    if m["bold"] > 6:
        fails.append(f"B {m['bold']} bold spans (max 6)")
    if m["sdev"] < 12:
        fails.append(f"B sentence-length spread {m['sdev']:.1f} words (need 12)")
    if m["run"] > 5:
        fails.append(f"B run of {m['run']} same-length sentences (max 5)")
    if m["longest"] > 200:
        fails.append(f"B longest paragraph {m['longest']} words (max 200)")
    if m["clip"] > 3:
        fails.append(f"B {m['clip']} consecutive one-sentence paragraphs (max 3)")
    ww = section(p, "Where we are")
    first = paragraphs(ww)[0] if paragraphs(ww) else ""
    two = " ".join(sentences(first)[:2])
    if not re.search(r"\byou\b|\byour\b", two, re.I):
        fails.append("B 'Where we are' does not address you in its first two sentences")
    if num > 1 and not re.search(r"\b(?:Chapter|Part) \d+\b|\bPart [1-8]\b|last chapter|previous chapter", first):
        fails.append("B 'Where we are' first paragraph does not name the previous chapter or part")
    lines = md.rstrip().split("\n")
    body_lines = lines[:-1] if lines[-1].startswith("Next:") else lines
    closing = [x for x in paragraphs("\n".join(body_lines)) if x]
    if closing:
        n = len(sentences(closing[-1]))
        if n != 1:
            fails.append(f"B closing paragraph has {n} sentences (need 1)")
    return fails

def check_manifest(path):
    fails = []
    for i, l in enumerate(open(path), 1):
        m = re.match(r'\s*\["(\d\d-[a-z0-9-]+)", "([^"]+)", "([^"]+)"\]', l)
        if m and re.match(IMPERATIVE, m.group(3)):
            fails.append(f"D blurb {m.group(1)} starts with an imperative: '{m.group(3)[:50]}'")
    return fails

def main(argv):
    report = "--report" in argv
    files = [a for a in argv if a != "--report"]
    bad = 0
    if report:
        print(f"{'ch':>3} {'words':>5} {'you/k':>5} {'bold':>4} {'lbl':>3} {'bul%':>4} {'sdev':>5} {'run':>3} {'long':>4} {'clip':>4}")
    for f in files:
        if f.endswith(".mjs"):
            fails = check_manifest(f)
        elif report:
            m = measure(open(f).read())
            print(f"{os.path.basename(f)[:2]:>3} {m['words']:>5} {m['youk']:>5.1f} {m['bold']:>4} {m['lbl']:>3} {m['bulpct']:>4.0f} {m['sdev']:>5.1f} {m['run']:>3} {m['longest']:>4} {m['clip']:>4}")
            continue
        else:
            fails = check_chapter(f)
        print(f"{os.path.basename(f)}: {len(fails)} failure(s)")
        for x in fails:
            print("  " + x)
        bad += len(fails)
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
