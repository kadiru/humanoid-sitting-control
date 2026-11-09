from pathlib import Path
import json,hashlib,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'progress_report_355';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
original=read(O/'SHA256.json')
for n,h in original.items():assert sha(O/n)==h
r=read(O/'path_351.json');p=r['dense']['path'];c=r['dense']['command_nodes'];plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(2,2,figsize=(14,10),layout='constrained')
axs[0,0].plot([e['clearance_m']*1000 for e in p]);axs[0,0].axhline(3,ls='--',color='#b84d43');axs[0,0].set(ylabel='Millimetres',title='Housing clearance\nUnchanged 3mm screen')
axs[0,1].plot([e['min_command_margin_rad'] for e in c]);axs[0,1].axhline(.03,ls='--',color='#b84d43');axs[0,1].set(ylabel='Radians',title='Static command margin\nUnchanged 0.03rad screen')
axs[1,0].plot(np.arange(90,105),[e['pelvis_gap_m']*1000 for e in p[90:105]],marker='.');axs[1,0].axhline(12,ls='--',color='#b84d43',label='12mm hover');axs[1,0].set(ylabel='Millimetres',title='Modeled hover gap\nEquality at knot99 remains violated');axs[1,0].legend()
axs[1,1].plot([e['foot_position_error_m']*1000 for e in p],label='Position error (mm)');axs[1,1].plot([np.rad2deg(e['foot_orientation_error_rad']) for e in p],label='Orientation error (deg)');axs[1,1].set(title='Full sampled foot geometry\nDifferent units shown separately in legend');axs[1,1].legend()
for ax in axs.flat:ax.set_xlabel('Knot index')
new=O/'geometry_and_headroom_review.png';assert not new.exists();fig.savefig(new,dpi=160);plt.close(fig)
s=(O/'findings.html').read_text();review=s.replace('src="geometry_and_headroom.png"','src="geometry_and_headroom_review.png"');assert review!=s and review.replace('src="geometry_and_headroom_review.png"','src="geometry_and_headroom.png"')==s;(O/'review.html').write_text(review)
for n,h in original.items():assert sha(O/n)==h
(O/'review_audit.json').write_text(json.dumps(dict(original_hashes_unchanged=True,change='Supplemental plot wraps titles and improves spacing; same recorded series. Review changes only one image-src attribute. Original HTML/plots/data unchanged.',new_calculations=False),indent=2));(O/'review_SHA256.json').write_text(json.dumps({n:sha(O/n) for n in ['geometry_and_headroom_review.png','review.html','review_audit.json']},indent=2));print('Review files written; original artifacts intact.')
