import json,re,html,struct,hashlib,concurrent.futures
from pathlib import Path
from collect import ROOT,fetch

ASSETS=[('1kwsjhpY84','L Couch'),('6MoOyPtetL','Couch Large'),
 ('mWgQ94zhDZ','Couch Medium'),('fFfoi1LNKY','Couch Medium'),
 ('l84E3NzGSF','Sofa'),('myd1WSucAz','Armchair')]

def get(asset):
    aid,name=asset
    url='https://poly.pizza/m/'+aid
    p=ROOT/'metadata'/f'poly_pizza_{aid}.html'
    if not p.exists():p.write_bytes(fetch(url))
    t=p.read_text()
    assert 'Public Domain (CC0)' in t
    img=html.unescape(re.search(r'property="og:image" content="([^"]+)"',t).group(1))
    glb=html.unescape(re.search(r'https://static\.poly\.pizza/[^"<>& ]+\.glb',t).group(0))
    preview=ROOT/'previews'/f'{aid}.jpg'
    if not preview.exists():preview.write_bytes(fetch(img))
    model=ROOT/'models'/f'{aid}.glb'
    if not model.exists():model.write_bytes(fetch(glb))
    b=model.read_bytes()
    magic,version,length=struct.unpack('<4sII',b[:12]);assert magic==b'glTF' and version==2 and length==len(b)
    json_len,json_type=struct.unpack('<I4s',b[12:20]);assert json_type==b'JSON'
    data=json.loads(b[20:20+json_len]);assert data['meshes']
    return dict(asset_id=aid,name=name,source='Poly Pizza',authors=['CreativeTrio' if aid=='myd1WSucAz' else 'Quaternius'],
      source_url=url,license='CC0',license_url=url,formats=['glb','fbx'],
      preview_url=img,preview_local=str(preview.relative_to(ROOT)),preview_sha256=hashlib.sha256(preview.read_bytes()).hexdigest(),
      download_check='Complete GLB retrieved; glTF header and mesh metadata parsed',
      gltf_url=glb,model_local=str(model.relative_to(ROOT)),model_sha256=hashlib.sha256(b).hexdigest(),model_bytes=len(b),
      dimensions_mm=None,mesh_count=len(data['meshes']))

if __name__=='__main__':
    (ROOT/'models').mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(get,ASSETS))
    (ROOT/'poly_pizza_selection.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({'models':len(rows),'total_glb_bytes':sum(r['model_bytes'] for r in rows)}))
