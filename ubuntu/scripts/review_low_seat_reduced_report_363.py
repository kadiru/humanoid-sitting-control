"""Whitespace-only readable companion; preserve original HTML and numerical tokens."""
from pathlib import Path
import re, hashlib, json

def main():
    d=Path('/data/humanoid-sitting-perception/low_seat_contact_01/progress_report_363')
    original=(d/'findings.html').read_text()
    parts=re.split(r'(<[^>]*>)',original)
    number=r'(?:\d*\.\d+|\d+)(?:[eE][+\-−]?\d+)?'
    for i in range(0,len(parts),2):
        if i and parts[i-1].startswith('<style'):continue
        text=parts[i]
        tokens=list(re.finditer(number,text)); out=[]; start=0
        for m in tokens:
            out.append(text[start:m.start()])
            if m.start() and text[m.start()-1].isalpha():out.append(' ')
            out.append(m.group())
            if m.end()<len(text) and text[m.end()].isalpha():out.append(' ')
            start=m.end()
        out.append(text[start:]);parts[i]=''.join(out)
    readable=''.join(parts)
    assert re.sub(r'\s','',original)==re.sub(r'\s','',readable)
    assert re.findall(r'<[^>]*>',original)==re.findall(r'<[^>]*>',readable)
    assert re.findall(number,original)==re.findall(number,readable)
    target=d/'readable.html';assert not target.exists();target.write_text(readable)
    note=d/'readability_review.json';note.write_text(json.dumps(dict(
        changes='Text-node spacing between words and numeric tokens only; scientific exponent tokens remain intact.',
        original_unchanged=True,canonical_nonwhitespace_exact=True,numeric_tokens_exact=True,
        tags_attributes_exact=True,original_sha256=hashlib.sha256((d/'findings.html').read_bytes()).hexdigest()),indent=2))
    (d/'review_SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [target,note]},indent=2))
    print('Readable companion created; original HTML, plots and data unchanged.')

if __name__=='__main__':main()
