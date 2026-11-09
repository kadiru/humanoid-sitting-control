"""Report completed read-only provenance work; no motion or physics claims."""
from pathlib import Path
import json,hashlib,html,shutil,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'progress_report_328'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
esc=lambda x:html.escape(str(x))
def main():
 O.mkdir(exist_ok=False)
 r=read(D/'prepared_contact_trace_v2_326/results.json');a=read(D/'prepared_contact_audit_327/independent_audit.json')
 assert a['passed'] and not r['physical_admission']
 for source,name in [(D/'prepared_contact_trace_v2_326/results.json','results.json'),
  (D/'prepared_contact_trace_v2_326/asset_inventory.json','asset_inventory.json'),
  (D/'prepared_contact_audit_327/independent_audit.json','independent_audit.json'),
  (D/'prepared_asset_scope_recovery_326/recovery.json','recovery.json'),
  (D/'prepared_contact_source_326/source_manifest.json','source_manifest.json')]:shutil.copy2(source,O/name)
 rows=''.join('<tr><th>'+esc(x['parameter'])+'</th><td>'+esc(x['supplied'])+'</td><td>'+esc(x['source_rule'])+'</td><td>'+esc(x['live'])+'</td></tr>' for x in r['rows'])
 ev=''.join('<tr><td>'+esc(x['file'])+':'+str(x['line'])+'</td><td><code>'+esc(x['token'])+'</code></td></tr>' for x in r['evidence'])
 urls={x['local']:x.get('url') for x in read(D/'prepared_contact_source_326/source_manifest.json')['sources']}
 links=' · '.join('<a href="'+url+'">'+esc(name)+'</a>' for name,url in urls.items() if url)
 summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),best_clean=64,development_cases=72,
  best_stable=70,remaining_slipping=8,remaining_housing_cases=2,new_physics_ticks=0,new_optimizer_updates=0,
  repair_complete=False,main_ppo_started=False,holdout_started=False,video_unchanged=True,
  source_files=len(urls),audit_checks=a['checks'],expanded_prims=378,instance_proxies=208,collision_prims=38)
 (O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 h='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Contact parameter provenance — round 328</title>
 <style>body{font:16px/1.55 system-ui,sans-serif;color:#182638;background:#f2f5f7;margin:0}main{max-width:1450px;margin:auto;padding:36px}h1{font-size:34px;line-height:1.15}h2{margin-top:42px}a{color:#165a98}table{border-collapse:collapse;width:100%;background:white}th,td{padding:14px;border:1px solid #cdd5df;text-align:left;vertical-align:top}th{background:#e9eef4}code{white-space:normal;overflow-wrap:anywhere;font-size:13px}.banner{background:#fff0cc;border-left:5px solid #b17d00;padding:18px}.metrics,.flow{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}.card{background:white;border:1px solid #b6c6d4;border-radius:9px;padding:18px;flex:1;min-width:200px}.card strong{display:block;font-size:24px}.unknown{border-color:#c58d4c;background:#fff7ee}video{width:100%;max-height:650px;background:#101b28}.small{font-size:14px;color:#536170}li{margin:8px 0}</style><main>
 <p class="small">Low-seat development evidence · source and configuration trace 326v2 / independent audit 327</p>
 <h1>Contact parameters are traced; internal stages remain unavailable</h1>
 <p class="banner">Physical repair remains unresolved. No new motion, controller, solver setting, reward, threshold or material change. No PPO, fresh holdout or policy promotion.</p>
 <div class="metrics"><div class="card"><strong>64 / 72</strong>Best verified clean trials; all 64 protected</div><div class="card"><strong>8 unresolved</strong>Remaining slipping cases; 2 include housing contact</div><div class="card"><strong>0 new physics ticks</strong>Read-only asset/source inspection</div></div>
 <h2>What the evidence can establish</h2><div class="flow"><div class="card"><strong>Inputs</strong>Installed schema defaults, preserved asset opinions, configured materials and captured stage attributes</div><div class="card unknown"><strong>Prepared rows</strong>Effective CFM, response, pair rest distance, bias and friction patches were not captured</div><div class="card unknown"><strong>Internal iterations</strong>Stage velocities, impulses, ordering and effective island counts were not captured</div><div class="card"><strong>Saved endpoints</strong>Joint/link states and aggregate contact-query returns; exact prefix replay already verified</div></div>
 <p>This source trace supplies provenance, not the missing solver history. End-of-step forces and reported gaps do not uniquely reconstruct prepared contact rows or internal iterations. No parameter was fitted to the endpoints.</p>
 <h2>Parameter inventory</h2><table><thead><tr><th>Parameter</th><th>Available inputs</th><th>Pinned source path</th><th>Captured status and limits</th></tr></thead><tbody>'''+rows+'''</tbody></table>
 <h2>A concrete provenance check</h2><p>The preserved robot asset authors stabilization <b>True</b>, while the saved instantiated scene reports <b>False</b>. Neither file changed. This illustrates why a default or authored asset opinion cannot be relabeled as an effective live value. It does not identify the cause of the raw-velocity discrepancy.</p>
 <h2>Asset coverage and preserved recovery</h2><p>Expanded inspection covers <b>378 prims, including 208 instance proxies, 44 rigid bodies and all 38 collision prims</b>. No authored CFM-scale, contact-offset, rest-offset or friction-combine opinion was found. The original default traversal inspected 170 prims and only 2 collision prims; its scope is marked incomplete and its source, log and results remain preserved. Version 2 adds instance-proxy traversal only. No physics or threshold changed.</p>
 <h2>Independent audit</h2><p>'''+str(a['checks'])+''' checks passed: recursive instance traversal, all asset paths/opinions, separate Sdf schema-default reads, exact saved live-state coverage, source hashes and line references. This is a provenance audit, <b>not physical admission or model validation</b>.</p>
 <h2>Control decision</h2><p>The existing evidence cannot recover the exact internal stages. No CFM/offset/iteration change or reconstructed velocity is admitted. Keep the unchanged 25-current-sample raw-velocity, pitch, gyro, geometry and support gates. A separately bounded alternate complete approach remains open; the missing internal state does not prove repair impossible. Its whole path, current-input reconstruction, inherited filter/rate, parent composition and measured support must pass preparation before physical validation. No specific alternate motion is certified by this report.</p>
 <h2>Source/version limits</h2><p>Public NVIDIA source is pinned to commit <code>fc1018a3745664a1db2b95ce03fb5e91eb585f2e</code> (110.0 / PhysX 5.8). Installed extension 110.3.2 is not proven binary-equivalent. The articulation branch is traced separately from the rigid-body-only preparation.</p><p>'''+links+'''</p><table><thead><tr><th>File and line</th><th>Verified source anchor</th></tr></thead><tbody>'''+ev+'''</tbody></table>
 <h2>Physical and effort results remain unchanged</h2><p>Latest full-20-second probe 273 is rejected: 20 stable, 14 clean, 8 slipping and 2 hip-housing/physical failures. Its selected 36 cm / +15° cases hover with zero legal seat load, seated hold and quiet release. No-slip hovering is not sitting. No new 72-case retention or effort comparison occurred.</p><p>All effort remains approximate implicit-PD telemetry, not isolated solver/hardware motor effort or energy. No failed hover, prefix or fragment is credited as a saving. Reference-assisted rigid-slab simulation does not establish unassisted, real-robot or anywhere sitting/leaning.</p>
 <h2>Complete attempt video remains current</h2><p>58.4 seconds · 44 recorded sources · 864 complete trials (503 clean / 361 failed), 192 diagnostic prefixes and 120 aborted case attempts without saved motion explicitly disclosed. Historical repeated totals are not controller accuracy. Recorded link-pose schematic, not camera footage. No new motion was added, so the original video was not re-rendered.</p><video controls preload="metadata" src="../all_attempts_video_307/all_attempts.mp4"></video>
 <p><a href="results.json">Trace data</a> · <a href="asset_inventory.json">Expanded asset inventory</a> · <a href="independent_audit.json">Independent audit</a> · <a href="recovery.json">Preserved scope correction</a> · <a href="source_manifest.json">Source hashes</a></p></main></html>'''
 (O/'findings.html').write_text(h)
 (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2)+'\n')
 print(json.dumps(summary))
if __name__=='__main__':main()
