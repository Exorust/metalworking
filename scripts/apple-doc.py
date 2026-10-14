#!/usr/bin/env python3 -I
"""Read a developer.apple.com/documentation page as text via Apple's JSON API.

usage: apple-doc.py <path-or-url> [--links]
  path examples: metal, metal/mtlcomputecommandencoder, metalperformanceshaders
  --links  also print the sub-pages (topic sections) with their URLs
Exit code 2 and 'HTTP <code>' if the page does not exist (broken link).
"""
import json, sys, urllib.request, urllib.error

def inline(parts):
    out = []
    for p in parts:
        t = p.get('type')
        if t == 'text': out.append(p.get('text', ''))
        elif t == 'codeVoice': out.append('`' + p.get('code', '') + '`')
        elif t in ('reference',): out.append(p.get('identifier', '').split('/')[-1])
        elif t in ('emphasis', 'strong'): out.append(inline(p.get('inlineContent', [])))
        elif t == 'link': out.append(p.get('title', p.get('destination', '')))
    return ''.join(out)

def blocks(content, depth=0):
    for b in content:
        t = b.get('type')
        if t == 'paragraph': print(inline(b.get('inlineContent', [])) + '\n')
        elif t == 'heading': print('#' * (b.get('level', 2)) + ' ' + b.get('text', '') + '\n')
        elif t == 'codeListing': print('```\n' + '\n'.join(b.get('code', [])) + '\n```\n')
        elif t == 'aside':
            print(f"[{b.get('style','note')}]"); blocks(b.get('content', []), depth)
        elif t in ('unorderedList', 'orderedList'):
            for item in b.get('items', []):
                sys.stdout.write('- '); blocks(item.get('content', []), depth + 1)
        elif t == 'table':
            for row in b.get('rows', []):
                print(' | '.join(inline(c[0].get('inlineContent', [])) if c and c[0].get('type')=='paragraph' else '' for c in row))
            print()
        elif t == 'termList':
            for it in b.get('items', []):
                print('- ' + inline(it.get('term', {}).get('inlineContent', [])) + ': ', end=''); blocks(it.get('definition', {}).get('content', []), depth+1)
        elif 'content' in b: blocks(b['content'], depth)
        elif 'inlineContent' in b: print(inline(b['inlineContent']) + '\n')

def main():
    arg = sys.argv[1]
    path = arg.replace('https://developer.apple.com/documentation/', '').replace('https://developer.apple.com/', '').strip('/')
    if path.startswith('documentation/'): path = path[len('documentation/'):]
    url = f'https://developer.apple.com/tutorials/data/documentation/{path}.json'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        print(f'HTTP {e.code} for https://developer.apple.com/documentation/{path}'); sys.exit(2)
    md = d.get('metadata', {})
    print(f"# {md.get('title')}  ({md.get('roleHeading','')})  URL: https://developer.apple.com/documentation/{path}")
    plats = ', '.join(f"{p.get('name')} {p.get('introducedAt','')}{' (beta)' if p.get('beta') else ''}{' deprecated' if p.get('deprecated') else ''}" for p in md.get('platforms', []) or [])
    if plats: print(f'Availability: {plats}')
    if d.get('abstract'): print('\n' + inline(d['abstract']) + '\n')
    for sec in d.get('primaryContentSections', []):
        k = sec.get('kind')
        if k == 'declarations':
            for dec in sec.get('declarations', []):
                print('```\n' + ''.join(t.get('text', '') for t in dec.get('tokens', [])) + '\n```\n')
        elif k == 'content': blocks(sec.get('content', []))
        elif k == 'parameters':
            for p in sec.get('parameters', []):
                print('- ' + p.get('name', '') + ': ', end=''); blocks(p.get('content', []))
    refs = d.get('references', {})
    if '--links' in sys.argv:
        for ts in d.get('topicSections', []):
            print(f"\n## Topic: {ts.get('title')}")
            for ident in ts.get('identifiers', []):
                r = refs.get(ident, {})
                u = r.get('url', '')
                print(f"- {r.get('title','?')}  https://developer.apple.com{u}  ::  {inline(r.get('abstract', []))[:160]}")
        for ss in d.get('seeAlsoSections', []):
            print(f"\n## See also: {ss.get('title')}")
            for ident in ss.get('identifiers', []):
                r = refs.get(ident, {}); print(f"- {r.get('title','?')}  https://developer.apple.com{r.get('url','')}")

main()
