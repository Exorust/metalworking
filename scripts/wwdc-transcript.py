#!/usr/bin/env python3 -I
"""Print the transcript of a WWDC session page. usage: wwdc-transcript.py https://developer.apple.com/videos/play/wwdc2026/330/"""
import re, html, sys, urllib.request
url = sys.argv[1]
try:
    h = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30).read().decode()
except Exception as e:
    print('FETCH FAILED', url, e); sys.exit(2)
t = re.search(r'<title>([^<]*)</title>', h)
print('#', html.unescape(t.group(1)) if t else url, '\nURL:', url)
m = re.search(r'id="transcript-content"(.*?)</section>', h, re.S)
if not m:
    print('NO TRANSCRIPT FOUND (page fetched, %d bytes)' % len(h)); sys.exit(1)
sents = re.findall(r'<span[^>]*class="sentence"[^>]*>(.*?)</span>', m.group(1), re.S)
print(' '.join(html.unescape(re.sub(r'<[^>]+>', '', s)).strip() for s in sents))
