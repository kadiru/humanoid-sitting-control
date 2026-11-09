from pathlib import Path
import json, shutil, hashlib
from PIL import Image
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path('/Users/kadir/humanoid-sitting-control')
OUT = ROOT / 'interview_report_20261022'
ASSETS = OUT / 'assets'
checkpoint = json.loads((ROOT/'validated_low_seat_contact_01/current_successor_checkpoint.json').read_text())
assert checkpoint['physical']['clean'] == '64/72'
assert checkpoint['last_accepted_research_iterate']['physical_success'] is False

# Figures are copied from recorded evidence or plotted from published summaries.
# No simulator, scientific solver, or model is called to prepare this report.
shutil.copyfile(ROOT/'validated_low_seat_contact_01/dual_view_audit_39/recorded_depth.png', ASSETS/'recorded_depth.png')
frame = Image.open(ASSETS/'historical_trials_frame.png')
# Coordinates below refer to the original 3840 x 2160 video frame at 3.9 seconds.
# Keep each selected tile intact, including its experiment and condition labels.
tiles = [
    (1912, 378, 2378, 660), # Ret09 old success
    (16, 90, 482, 372),    # 164 measured balance failure
    (490, 90, 956, 372),   # 150 balanced forces failure
    (16, 1242, 482, 1524), # 273 hovering failure
]
tile_w, tile_h, gap = 700, 424, 18
gallery = Image.new('RGB', (tile_w*2+gap, tile_h*2+gap), 'white')
for i, box in enumerate(tiles):
    tile=frame.crop(box)
    tile=tile.resize((tile_w,tile_h), Image.Resampling.LANCZOS)
    x=(i%2)*(tile_w+gap); y=(i//2)*(tile_h+gap)
    gallery.paste(tile,(x,y))
gallery.save(ASSETS/'selected_recorded_trials.png')

pages=[]
def page(title):
    blocks=[];pages.append({'title':title,'blocks':blocks});return blocks
def p(b,t): b.append(('p',t))
def h(b,t): b.append(('h',t))
def table(b,heads,rows,widths): b.append(('table',heads,rows,widths))
def image(b,name,caption,width=6.8): b.append(('image',name,caption,width))
def bullets(b,items):b.append(('bullets',items))

b=page('Humanoid sitting control research report')
p(b,'Project evidence and interview preparation for Kadir\nSnapshot: October 22, 2026 Pacific time / October 23, 2026 UTC')
p(b,'We are developing and evaluating a humanoid sitting controller in simulation, with the current work focused on low seats that slope toward the feet. The aim is to establish stable, legal seat contact without slipping or unwanted robot-housing contact, preserve all existing successes, and then compare a repaired controller with new reinforcement-learning policies, measuring success before actuator effort.')
p(b,'The project has a functioning baseline, a measured-sensor control pipeline, and a detailed experimental record. The remaining repair is unfinished: the best controller passes 64 of 72 development trials, and none of the eight remaining failures has been repaired. Recent numerical progress has not yet passed the checks required to execute a new motion. [S1, S2]')
table(b,['Measure','Verified position'],[
('Clean complete trials','64 / 72 = 88.9%'),
('Stable trials','70 / 72; stability alone does not imply a clean trial'),
('Slipping and housing contact','8 slipping; 2 housing-contact cases; categories overlap'),
('Remaining repair','0 / 8 failing trials repaired; all 64 successes protected'),
('New main PPO comparison','Not started'),
('Fresh evaluation bank','24 reserved trials; unopened and unavailable for tuning'),
('Latest research cycle','Closed after a failed construction; no new controller accepted'),
], [2.25,4.55])
h(b,'What completion means')
p(b,'The required outcome is a repaired controller verified across all 72 complete development trials, followed by the specified new PPO comparison and reserved evaluation, plus an updated video covering the attempts in less than one minute. Passing the development bank would establish success on that bank; it would not establish general sitting ability or real-robot deployment.')
p(b,'The eight failures are four surface conditions in two placements each: 36 cm / +15°, 36 cm / +20°, 37 cm / +16.25°, and 36.5 cm / +17.5°. The 88.9% figure is a test pass rate, not a percentage of project completion. [S2]')

b=page('System architecture and tools')
p(b,'The simulation uses the G1 humanoid asset in Isaac Lab. The approach combines geometric motion planning, measured feedback, guarded contact transfer, and previously learned policy components. Earlier PPO work learned bounded corrections around a frozen motion scaffold; it was not learning sitting entirely from scratch. [S7]')
p(b,'Runtime flow: measured depth and proprioception → seat geometry and state estimates → motion reference, feedback and learned corrections → guards and filtered joint commands → simulated actuators and contact. Offline optimization proposes future motion updates; it does not directly replace this guarded pipeline.')
table(b,['Tool or component','Role in the work'],[
('Isaac Lab and PhysX','Robot articulation, joint actuators, rigid contact, cameras and parallel simulated environments.'),
('Python and NumPy','Experiment scripts, geometry, recorded arrays, evaluation and reproducible data processing.'),
('PyTorch and RSL-RL','Learned policies and PPO infrastructure; PyTorch also differentiates the offline mechanics model.'),
('SciPy and HiGHS','Sparse matrix construction, rotations and bounded linear-programming experiments, including interior-point solves.'),
('Exact and finite arithmetic checks','Rational algebra for specific transformations; separate binary64 and float32 execution checks.'),
('Mac plus Ubuntu over SSH','Local review and artifacts; direct execution on the Ubuntu workstation in the project’s Python environment.'),
('JSON, NPZ, CSV and SHA256','Saved inputs, sources, outcomes, manifests and independent audit records.'),
('Matplotlib, HTML and FFmpeg','Diagnostic plots, review reports and video from recorded trajectories.'),
('Codex assistance','Source inspection, implementation, analysis and reporting under user direction; one scientific execution owner.'),
],[1.93,4.87])
p(b,'PPO means proximal policy optimization; IMU means inertial measurement unit; PD means proportional–derivative joint control. Proprioception refers here to joint and body-motion measurements.')
p(b,'Runtime decisions use measured camera depth, encoder, IMU and torque-equivalent signals, together with a static robot model and reference. Privileged simulator information is restricted to reset, reward and evaluation. This information boundary is central to the experiment, although hardware transfer remains untested. [S2, S7]')

b=page('Perception and modeling findings')
h(b,'Recovering seat measurements when the rear view is lost')
p(b,'The rear camera can lose a usable seat-plane fit during the motion. A recorded diagnostic checked the already defined downward camera as an additional view. It produced valid fits for all 132 targeted frames in which the rear tracker was unavailable; the largest recorded height and slope errors there were 1.565 mm and 0.132°. These are results for that diagnostic, not universal perception accuracy. [S3]')
image(b,'recorded_depth.png','Figure 1. Actual recorded simulated depth at 12 seconds from diagnostic 39. The rear view has no fit in these examples; the downward view does. This recording itself did not control the robot. A subsequent revision introduced a guarded current-frame fallback. [S3]',6.8)
p(b,'The runtime change retained the rear view as preferred and allowed the current downward fit only for selected cases when the rear fit was unavailable. Plane conditioning, self masking, floor exclusion and patch-association rules stayed in place. This addressed a specific observation gap without substituting simulator geometry for measured input.')
h(b,'Finding a mismatch between planning and collision geometry')
p(b,'PhysX convex cooking changed the pelvis collision support surface by about 5–6 mm in recorded contact directions. A path checked against the raw mesh could therefore misestimate clearance. The project introduced a separate offline cooked collision model for planning while retaining the original geometry for perception self masking. This improved model fidelity; it did not certify a successful contact transition. [S3]')
h(b,'Aligning the planned endpoint with the actual objective')
p(b,'Some earlier paths targeted a 0.9 radian trunk tilt, about 51.6°, even though the resting criterion required at most 40°. The earlier model screens were incomplete. Subsequent planning used the unchanged 40° requirement with a separate 2° research reserve. The correction illustrates why the acceptance contract must be checked before optimizing a trajectory. [S3]')

b=page('Physical experiments and what they taught us')
image(b,'selected_recorded_trials.png','Figure 2. Four intact tiles selected from the historical video at 3.9 seconds, showing saved poses at trial time 17.6 seconds and full-trial verdicts. The green tile is a Ret09 success at 36 cm / +5°; the red tiles are different experiments at 36 cm / +15°. These are schematic renderings from recorded simulation transforms, not camera photographs or a matched causal comparison. [S8]',6.8)
h(b,'A smoother model did not guarantee a successful landing')
p(b,'A force-allocation diagnostic found a 0.0795 radian command jump for only 0.00087 radians of pose change. A revised allocation removed that initial jump and reduced adjacent modeled ankle-torque reversals from roughly 26–27 Nm to below 0.5 Nm. The later physical probe still failed. Improving one numerical artifact was useful, but balance, tracking and contact transfer remained coupled problems. [S3]')
h(b,'Hovering without slipping is not sitting')
p(b,'The most recent physical candidate recorded in the handoff, experiment 273, could hover without establishing the required legal seat load and quiet release. It was rejected. A clean outcome requires complete-trial slip and contact checks, stable support and the resting definition, including trunk tilt at most 40° and legal seat normal load of at least 5% of body weight. [S2]')
p(b,'The historical 58.4-second video covers 864 complete trials across multiple experiments, 192 recorded prefixes, and 120 aborted diagnostic cases disclosed without invented motion. Those totals are an experiment archive, not additional independent successes for the current baseline. The video has not been updated with a repaired controller or the planned new PPO comparison. [S8]')

b=page('Recent optimization progress and failure')
p(b,'Work shifted toward constructing a complete motion that satisfies foot pose, balance, housing clearance, joint bounds, command filtering and rates, contact transfer and timing together. One full linear representation has 78,791 variables and 652,529 inequalities. A candidate vector must pass the original constraints independently, then survive finite execution and nonlinear model checks before any simulated motion. [S1, S4]')
table(b,['Saved nonlinear model result','Value'],[('Mismatch merit before update','0.1979'),('Mismatch merit after update','0.1227'),('Relative reduction','38.0%'),('Actual motion admission','Failed')],[4.7,2.1])
p(b,'Mismatch merit is a normalized model measure. Its reduction is not a physical success-rate improvement, energy saving or controller-performance gain. [S4]')
p(b,'The update passed finite encoding and its independent audit. Fresh model and derivative evaluations covered 1,365 states in 8.27 seconds total. It was accepted as a research iterate, but actual numerical motion admission failed: residual mismatch included about 7.75 mm in foot position, 0.0266 rad in commands and 2.63 Nm in torque. No repaired motion followed. [S4]')
h(b,'Why reducing the problem size was insufficient')
p(b,'An earlier exact elimination reduced the optimizer to 10,541 coordinates but made its inequalities much denser, with about 6.10 million nonzeros. A cold HiGHS solve returned no candidate after six interior-point iterations and a charged total of 45.683 seconds, exceeding the 45-second allowance. Preparation and validation costs matter alongside solver time; fewer variables alone did not establish a speedup. [S2]')
h(b,'Outcome of the latest approved research cycle')
p(b,'Stage A identified the old correction’s obstruction: a right-ankle-roll planning lower bound at 13.14 seconds, with only about 2.50 × 10⁻¹⁰ rad of guarded margin. Stage B held that ankle coordinate fixed and released the existing pelvis-orientation freedom, but its predicted improvement was only 4.46 × 10⁻¹³ and failed the numerical progress floor. The cycle stopped there; later stages, simulation and PPO did not start. The audit verified the failure accounting, not a feasible candidate. [S5]')
p(b,'This is a negative result for that construction. It does not prove that the full motion problem or low-seat sitting is impossible. The physical score remains 64/72.')

b=page('Remaining work and the next decision')
p(b,'The next milestone is one complete, admissible motion update that can advance to controlled simulation. There is currently no selected, launch-ready method. The latest local design note considers solving small state-local blocks inside the Newton system while retaining temporal coupling. It may reduce the size of the remaining global system, but the existing sparse solver may already exploit the same structure. Net speedup, numerical conditioning, initialization and complete resource costs are unresolved. [S6]')
table(b,['Stage','Required evidence','Current status'],[
('New construction','A concrete mechanism, original-constraint checks, independent audit, and complete runtime and memory accounting.','Design only'),
('Executable motion','Interior, finite encoding, nonlinear mechanics, geometry, command filtering, float32 runtime and both placements.','No current admitted repair'),
('Development repair','All 72 matched full 20-second trials; keep the 64 successes and remove all eight failures.','64/72; 0/8 repaired'),
('New PPO comparison','Two prescribed runs; validated channels, replay, optimizer changes and frozen accounting.','Not started'),
('Reserved evaluation','Freeze both final policies and development selections before opening the fresh 24 cases.','Closed'),
('Final communication','Complete results and an updated subminute video including new physical and PPO coverage.','Historical video only'),
],[1.3,3.95,1.55])
p(b,'A new method must fit the inherited 45-second total scientific limit, including validation and audit, and the 40 GiB per-worker cap. The completed one-time diagnostic exceptions are consumed. They do not authorize retries, relaxed tolerances, reuse of rejected directions, or automatic downstream runs. No new experiment was launched to prepare this report.')
h(b,'How the final comparison will be measured')
p(b,'After the repair passes, the plan specifies two PPO seeds, 63111 and 63117, with 24 environments and eight complete-episode updates per seed. Each run has 192,000 active plus 4,200 indexed warmup transitions, or 196,200 per seed and 392,400 total; initial settling is accounted separately. The main runs start from the smoke test’s initial weights. [S2]')
p(b,'Report clean support, stability, slip, unwanted contact and lost or gained successes before effort. Compare effort only on common complete clean trials, across the full 20 seconds and final 10- and 5-second windows. Applied PD torque is a simulation effort proxy, not measured electrical energy. The fresh bank contains four unseen seat heights crossed with three positive slopes in two placements. [S2]')

b=page('Interview discussion and lessons')
h(b,'A concise project description')
p(b,'“This project investigates reliable humanoid sitting using a G1 simulation, measured depth and proprioceptive inputs, geometric control and learned policy components. The current baseline succeeds cleanly in 64 of 72 complete development trials. The difficult remaining cases are low, positively sloped seats where balance, clearance and actuator limits interact. The work exposed concrete model errors, including a 5–6 mm collision-geometry discrepancy and a planned trunk angle that violated the resting objective. More recent offline work reduced nonlinear model mismatch by 38%, but it still failed motion-admission checks. The next objective is a validated repair that preserves all existing successes, followed by a frozen two-seed PPO comparison and reserved evaluation.”')
h(b,'Technical topics worth preparing')
bullets(b,[
    'Data boundaries: explain why the controller uses measured inputs while simulation truth is allowed for evaluation and reward; describe how leakage would invalidate the claim.',
    'Debugging across layers: trace a failure from a camera or collision model through planning, actuator filtering, tracking and contact. Use one recorded example rather than listing every experiment.',
    'Numerical engineering: explain why a smaller matrix can be slower, what an independently checked feasible vector means, and why model improvement does not imply safe motion.',
    'Experiment design: explain complete-trial metrics, overlapping failure categories, regression retention, frozen budgets and why the fresh evaluation set remains unopened.',
    'Failure and prioritization: discuss when to stop a formulation, what evidence a new approach must add, and how to measure progress toward a useful controller.',
])
h(b,'What the setbacks say about the process')
p(b,'There is a real risk of overinvesting in numerical certification while physical performance stays flat. A useful lesson is to distinguish essential physical and regression criteria from mathematical ambitions such as finding an optimum. Limited feasibility-only diagnostics were explicitly tried; the latest still failed. The next approach needs a credible path to a usable update, not another unchanged design note. The eight remaining physical failures are the primary progress measure.')
p(b,'Earlier PPO experience reinforces this point. In a separate reward study, two 128-update runs produced 44/64 and 42/64 successes versus 47/64 for the starting policy; neither was promoted. This is a different benchmark from the current 72-case bank. More training and a changed reward did not automatically improve the evaluated behavior. [S9]')
p(b,'Ownership note: this report describes project-level evidence from work conducted with AI coding assistance. Before using first-person interview answers, identify which architecture choices, code, analyses and experiments you personally performed or reviewed. The record does not establish sole authorship, team leadership, production deployment or business impact.')

b=page('ChatGPT preparation prompt and evidence')
h(b,'Prompt to paste with this report')
p(b,'I am preparing for a Meta interview. Use the attached project report as factual context. First ask me which role and level I am interviewing for and what I personally owned, implemented and reviewed. Then help me develop a 90-second summary, a five-minute technical explanation, and one STAR story about debugging or an unsuccessful experiment. Ask realistic follow-up questions about metrics, data leakage, control versus learning, numerical constraints, experiment design, tradeoffs and prioritization. Challenge unsupported claims. Distinguish simulation results, model-only improvements and proposed work. Do not invent real-robot deployment, completed PPO results, generalized performance, energy savings, team size or business impact. Help me explain what I would do differently and why the project remains unfinished.')
h(b,'Evidence behind the report')
p(b,'The source records below are local project artifacts. References use the project root /Users/kadir/humanoid-sitting-control. The plain-text companion contains the same report content, including descriptions of the figures, so it can be pasted without access to these files.')
sources=[
('S1','Current status and accepted model iterate','validated_low_seat_contact_01/current_successor_checkpoint.json'),
('S2','Goal, baseline, failure conditions and complete evaluation contract','task_transfer_2026-10-20/START_HERE.txt; task_transfer_2026-10-20/CONTINUE_IN_NEW_CHAT.txt'),
('S3','Perception, geometry and control findings','task_transfer_2026-10-20/evidence/LOW_SEAT_CONTACT_PROTOCOL.md; validated_low_seat_contact_01/dual_view_audit_39/recorded_depth.png'),
('S4','Full linear system and nonlinear model result','validated_low_seat_contact_01/changed_state_primary_20261020_v1/summary.json; validated_low_seat_contact_01/finite_model_step_20261020_v1/decision.json; summary.json'),
('S5','Latest failed cycle and its closure','validated_low_seat_contact_01/feasibility_research_cycle_result_20261022_v1.txt; feasibility_research_cycle_closure_20261022_v1.json'),
('S6','Unexecuted structural design and worker observation','validated_low_seat_contact_01/local_block_design_assessment_20261023_v1.json'),
('S7','Simulation, learning and differentiable mechanics sources','ubuntu_audit/sitting_task/src/humanoid_sitting/tasks/config.py; rl_support_work/scripts/train_support_rl.py; rl_support_work/scripts/low_seat_clock_transcription_444.py'),
('S8','Historical visual record and coverage manifest','validated_low_seat_contact_01/all_attempts_video_307/all_attempts.mp4; coverage.json'),
('S9','Separate earlier PPO reward study','validated_support_reward_study_01/results/REPORT.md'),
]
for sid,label,path in sources:
    b.append(('source',f'[{sid}] {label}',path))
p(b,'Snapshot precedence: the October 23 UTC successor checkpoint supersedes the historical checkpoint 603. The latest recorded worker observation found no scientific workers at 03:36 UTC. The overall task remains active; this is a report of saved evidence, not a new scientific run.')

doc=Document();section=doc.sections[0]
section.page_width=Inches(8.5);section.page_height=Inches(11)
section.top_margin=Inches(.7);section.bottom_margin=Inches(.65)
section.left_margin=Inches(.85);section.right_margin=Inches(.85)
section.header_distance=Inches(.27);section.footer_distance=Inches(.28)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
    s=doc.styles[name];s.font.name='Arial';s.font.color.rgb=RGBColor(0,0,0)
    for elem in list(s.element.xpath('.//w:color')):
        for attr in [qn('w:themeColor'),qn('w:themeTint'),qn('w:themeShade')]:elem.attrib.pop(attr,None)
    for elem in list(s.element.xpath('.//w:pBdr')):elem.getparent().remove(elem)
normal=doc.styles['Normal'];normal.font.size=Pt(10.5)
normal.paragraph_format.space_after=Pt(7);normal.paragraph_format.line_spacing=1.1
doc.styles['Title'].font.size=Pt(26);doc.styles['Title'].paragraph_format.space_after=Pt(9)
doc.styles['Heading 1'].font.size=Pt(20);doc.styles['Heading 1'].paragraph_format.space_after=Pt(12)
doc.styles['Heading 2'].font.size=Pt(12);doc.styles['Heading 2'].font.bold=True
doc.styles['Heading 2'].paragraph_format.space_before=Pt(10);doc.styles['Heading 2'].paragraph_format.space_after=Pt(5)
doc.styles['Caption'].font.size=Pt(8.5);doc.styles['Caption'].font.italic=False;doc.styles['Caption'].font.bold=False
doc.styles['Caption'].paragraph_format.space_after=Pt(8);doc.styles['Caption'].paragraph_format.line_spacing=1.05
hp=section.header.paragraphs[0];hp.text='HUMANOID SITTING CONTROL  |  PROJECT REPORT'
hp.runs[0].font.size=Pt(8);hp.runs[0].font.color.rgb=RGBColor(0,0,0)
fp=section.footer.paragraphs[0];fp.alignment=WD_ALIGN_PARAGRAPH.RIGHT
fp.add_run('October 22, 2026  •  ')
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');fp._p.append(fld)
for r in fp.runs:r.font.size=Pt(8)

def add_table(heads,rows,widths):
    t=doc.add_table(rows=1,cols=len(heads));t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for c,w in zip(t.columns,widths):c.width=Inches(w)
    for c,txt,w in zip(t.rows[0].cells,heads,widths):c.text=txt;c.width=Inches(w)
    repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in rows:
        for c,txt,w in zip(t.add_row().cells,row,widths):c.text=txt;c.width=Inches(w)
    for i,row in enumerate(t.rows):
        prevent=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(prevent)
        for j,c in enumerate(row.cells):
            c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcPr=c._tc.get_or_add_tcPr()
            borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
            tcPr.append(borders)
            mar=OxmlElement('w:tcMar')
            for name,v in [('top','85'),('bottom','85'),('left','100'),('right','100')]:
                el=OxmlElement('w:'+name);el.set(qn('w:w'),v);el.set(qn('w:type'),'dxa');mar.append(el)
            tcPr.append(mar)
            if i==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E8EEF2');tcPr.append(shade)
            for para in c.paragraphs:
                para.paragraph_format.space_after=Pt(0);para.paragraph_format.line_spacing=1.05
                for r in para.runs:r.font.size=Pt(9.3);r.bold=(i==0)
    doc.add_paragraph().paragraph_format.space_after=Pt(0)

text=[]
for pi,pg in enumerate(pages):
    if pi:doc.add_page_break()
    doc.add_paragraph(pg['title'],style='Title' if pi==0 else 'Heading 1')
    text += [pg['title'].upper(),'']
    for block in pg['blocks']:
        kind=block[0]
        if kind in ('p','h'):
            doc.add_paragraph(block[1],style='Heading 2' if kind=='h' else None)
            text += [block[1],'']
        elif kind=='bullets':
            for item in block[1]:
                para=doc.add_paragraph(item,style='List Bullet');para.paragraph_format.space_after=Pt(5)
                text.append('- '+item)
            text.append('')
        elif kind=='table':
            _,heads,rows,widths=block;add_table(heads,rows,widths)
            text += [' | '.join(heads)]+[' | '.join(row) for row in rows]+['']
        elif kind=='image':
            _,name,caption,width=block
            para=doc.add_paragraph();para.paragraph_format.space_after=Pt(3);para.paragraph_format.keep_with_next=True
            para.alignment=WD_ALIGN_PARAGRAPH.CENTER
            run=para.add_run();run.add_picture(str(ASSETS/name),width=Inches(width))
            props=run._r.xpath('.//wp:docPr')
            if props:props[0].set('descr',caption)
            doc.add_paragraph(caption,style='Caption')
            text += ['[Figure] '+caption,'']
        elif kind=='source':
            para=doc.add_paragraph();para.paragraph_format.space_after=Pt(3)
            r=para.add_run(block[1]);r.bold=True;r.font.size=Pt(8.8)
            para=doc.add_paragraph(block[2]);para.paragraph_format.space_after=Pt(6);para.paragraph_format.line_spacing=1.0
            for r in para.runs:r.font.size=Pt(8.1)
            text += [block[1],block[2],'']

doc.core_properties.title='Humanoid sitting control research report'
doc.core_properties.subject='Verified project evidence and interview preparation'
doc.core_properties.author='Prepared with Codex for Kadir'
doc.core_properties.keywords='humanoid robotics, simulation, controls, PPO, interview preparation'
doc.save(OUT/'Humanoid_Sitting_Project_Report.docx')
(OUT/'ChatGPT_Project_Brief.txt').write_text('\n'.join(text))
(OUT/'report_content.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False))
provenance={'scope':'Report preparation only; no new scientific execution','checkpoint_utc':checkpoint['utc'],
  'source_checkpoint_sha256':hashlib.sha256((ROOT/'validated_low_seat_contact_01/current_successor_checkpoint.json').read_bytes()).hexdigest(),
  'video_frame_seconds':3.9,'video_frame_selected_tile_boxes':tiles,'sources':[dict(id=s,label=l,path=p) for s,l,p in sources],
  'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'Humanoid_Sitting_Project_Report.docx',OUT/'ChatGPT_Project_Brief.txt']}}
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
print(json.dumps({'docx':str(OUT/'Humanoid_Sitting_Project_Report.docx'),'txt':str(OUT/'ChatGPT_Project_Brief.txt'),'words':len(' '.join(text).split()),'designed_pages':len(pages)}))
