"""Build the dark step-diff page of the report loop.

Run from report/source: `python3 diffhtml.py . ../report-loop-diffs.html`.
"""
import difflib, html, re, sys
S=sys.argv[1]; out=sys.argv[2]
def toks(t): return re.findall(r'\S+|\s+', t)
def render(a,b):
    A,B=toks(open(a).read()),toks(open(b).read())
    sm=difflib.SequenceMatcher(None,A,B,autojunk=False)
    parts=[];rem=add=0
    for op,i1,i2,j1,j2 in sm.get_opcodes():
        if op=='equal': parts.append(html.escape(''.join(A[i1:i2])))
        else:
            if i2>i1:
                d=''.join(A[i1:i2]); rem+=len(d.split()); parts.append(f'<del>{html.escape(d)}</del>')
            if j2>j1:
                s=''.join(B[j1:j2]); add+=len(s.split()); parts.append(f'<ins>{html.escape(s)}</ins>')
    total=len(''.join(B).split())
    return ''.join(parts),rem,add,total
tabs=[("Step 3: Gemini → fact-fixed","dba3803-report-gemini.md","dba3803-report-gemini-factfixed.md","Claude fixes only what is wrong (factcheck-1.md, 18 items)."),
      ("Step 3: Gemini's additions → fixed","gemini-additions.md","gemini-additions-factfixed.md","The pieces Gemini wrote for what was missing (gemini-brief-2.txt); Claude fixes only what is wrong (factcheck-3.md)."),
      ("Step 4: merged → plain words","dba3803-report-merged.md","dba3803-report-step4.md","Fact-fixed report with the fixed additions placed where Gemini said, then one-for-one plain-word swaps and fixes to bad sentences only (step4.py)."),
      ("Step 5: final check","dba3803-report-step4.md","dba3803-report-final.md","Last fact check: wrong claims fixed (factcheck-3.md).")]
secs=[];btns=[]
for k,(name,a,b,note) in enumerate(tabs):
    body,rem,add,total=render(f"{S}/{a}",f"{S}/{b}")
    btns.append(f'<button data-t="{k}"{" class=on" if k==0 else ""}>{html.escape(name)}</button>')
    secs.append(f'<section id="t{k}"{"" if k==0 else " hidden"}><p class=meta>{html.escape(note)} <b>{rem:,}</b> words removed, <b>{add:,}</b> added; result {total:,} words.</p><div class=doc>{body}</div></section>')
page=f'''<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Report loop diffs</title>
<style>:root{{color-scheme:dark}}body{{background:#16171a;color:#d8d8d8;font:15px/1.6 -apple-system,system-ui,sans-serif;margin:0;padding:16px}}
.wrap{{max-width:900px;margin:0 auto}}h1{{font-size:20px;margin:8px 0 12px}}nav{{position:sticky;top:0;background:#16171a;padding:8px 0;display:flex;gap:8px;flex-wrap:wrap;border-bottom:1px solid #2a2c31}}
button{{background:#24262b;color:#ccc;border:1px solid #33363c;border-radius:6px;padding:6px 10px;cursor:pointer;font:inherit;font-size:14px}}button.on{{background:#3a4a6b;color:#fff;border-color:#4d6390}}
.meta{{color:#9a9a9a;font-size:14px}}.doc{{white-space:pre-wrap;word-wrap:break-word}}del{{background:#4a1f22;color:#f0a0a0;text-decoration:line-through}}ins{{background:#1d3f26;color:#a6e3b0;text-decoration:none}}</style></head>
<body><div class=wrap><h1>DBA3803 report: what each loop step changed</h1><nav>{"".join(btns)}</nav>{"".join(secs)}</div>
<script>document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));document.querySelectorAll('section').forEach((s,i)=>s.hidden=i!=b.dataset.t);scrollTo(0,0)}})</script></body></html>'''
open(out,'w').write(page)
