"""Readable text-node spacing only; original report and hashes stay unchanged."""
from pathlib import Path
from html.parser import HTMLParser
import html,re,json,hashlib
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01/progress_report_v2_32')
class Spacing(HTMLParser):
 def __init__(self):super().__init__(convert_charrefs=True);self.parts=[];self.raw=False
 def handle_decl(self,x):self.parts.append('<!'+x+'>')
 def handle_starttag(self,tag,attrs):self.parts.append(self.get_starttag_text());self.raw=tag in ['style','script'] or self.raw
 def handle_startendtag(self,tag,attrs):self.parts.append(self.get_starttag_text())
 def handle_endtag(self,tag):self.parts.append('</'+tag+'>');self.raw=False if tag in ['style','script'] else self.raw
 def handle_data(self,data):
  if self.raw:self.parts.append(data);return
  data=re.sub(r'(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])',' ',data)
  data=re.sub(r',(?=\S)',', ',data);data=re.sub(r';(?=\S)','; ',data);data=re.sub(r':(?=\S)',': ',data)
  self.parts.append(html.escape(data,quote=False))
p=D/'findings.html';out=D/'review.html';assert not out.exists();parser=Spacing();parser.feed(p.read_text());out.write_text(''.join(parser.parts))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(D/'review_audit.json').write_text(json.dumps(dict(scope='Formatting-only text-node spaces; original findings and16-file SHA256 manifest preserved. No numerical/result/claim change.',original_findings_sha256=sha(p),review_sha256=sha(out),formatter_sha256=sha(Path(__file__))),indent=2))
print('Formatted review saved without changing original report.')
