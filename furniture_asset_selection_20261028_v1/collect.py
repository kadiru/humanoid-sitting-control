"""Collect public model manifests and publisher thumbnails for a local selection board.
Does not import anything into the simulator or execute downloaded content.
"""
import json, urllib.request, concurrent.futures, hashlib, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UA = 'RobotFurnitureSelection/1.0'
IDS = '''sofa_02 sofa_03 chinese_sofa painted_wooden_sofa
modern_arm_chair_01 mid_century_lounge_chair plastic_monobloc_chair_01
dining_chair_02 painted_wooden_chair_01 gallinera_chair
painted_wooden_bench wooden_stool_01 folding_wooden_stool metal_stool_02
wooden_stool_02 bar_chair_round_01
modern_coffee_table_01 modern_coffee_table_02 chinese_tea_table
coffee_table_round_01 industrial_coffee_table wooden_table_02
round_wooden_table_02 chinese_console_table'''.split()

def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':UA}),timeout=40).read()

def ph_item(asset_id):
    cat=json.loads((ROOT/'polyhaven_catalog.json').read_text())[asset_id]
    files_path=ROOT/'metadata'/f'{asset_id}.json'
    if not files_path.exists(): files_path.write_bytes(fetch(f'https://api.polyhaven.com/files/{asset_id}'))
    files=json.loads(files_path.read_text())
    image_url=cat['thumbnail_url'].replace('width=256','width=512').replace('height=256','height=512')
    image_path=ROOT/'previews'/f'{asset_id}.webp'
    if not image_path.exists(): image_path.write_bytes(fetch(image_url))
    formats=[k for k in ('blend','gltf','usd','fbx') if k in files]
    assert formats, asset_id
    entry=files['gltf']['1k']['gltf']
    # Verify that the actual mesh descriptor is retrievable; no dependency bulk-download.
    model_path=ROOT/'metadata'/f'{asset_id}_1k.gltf'
    if not model_path.exists(): model_path.write_bytes(fetch(entry['url']))
    gltf=json.loads(model_path.read_text())
    assert gltf.get('meshes'), asset_id
    return dict(asset_id=asset_id,name=cat['name'],source='Poly Haven',
        source_url=f'https://polyhaven.com/a/{asset_id}',authors=list(cat['authors']),
        license='CC0',license_url='https://polyhaven.com/license',formats=formats,
        dimensions_mm=cat.get('dimensions'),preview_url=image_url,
        preview_local=str(image_path.relative_to(ROOT)),
        preview_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        download_manifest=str(files_path.relative_to(ROOT)),
        gltf_url=entry['url'],download_check='glTF mesh descriptor retrieved and parsed',
        gltf_1k_total_bytes=entry['size']+sum(v['size'] for v in entry.get('include',{}).values()),
        mesh_count=len(gltf['meshes']))

if __name__=='__main__':
    (ROOT/'metadata').mkdir(exist_ok=True)
    (ROOT/'previews').mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(ph_item,IDS))
    (ROOT/'polyhaven_selection.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({'models':len(rows),'previews':len(rows),'mesh_descriptors_verified':len(rows)}))
    for name,url in [('quaternius_bundle','https://poly.pizza/bundle/Ultimate-House-Interior-Pack-2SXnFbwFzm'),('creative_armchair','https://poly.pizza/m/myd1WSucAz'),('simple_sofa','https://poly.pizza/m/l84E3NzGSF')]:
        html=fetch(url).decode()
        (ROOT/'metadata'/f'{name}.html').write_text(html)
        print(name, 'page bytes',len(html))
        for match in re.finditer(r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>',html,re.S):
            data=json.loads(match.group(1))
            (ROOT/'metadata'/f'{name}_data.json').write_text(json.dumps(data,indent=2))
            print('next data keys',data.keys())
