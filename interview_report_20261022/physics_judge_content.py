from pathlib import Path
import json, hashlib
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path('/Users/kadir/humanoid-sitting-control')
BASE=ROOT/'interview_report_20261022'
OUT=BASE/'physics_judge'
ASSETS=BASE/'assets'
pages=[]
def page(t):
    b=[];pages.append({'title':t,'blocks':b});return b
def p(b,t):b.append(('p',t))
def h(b,t):b.append(('h',t))
def bullets(b,items):b.append(('bullets',items))
def table(b,heads,rows,widths):b.append(('table',heads,rows,widths))
def image(b,name,caption,width):b.append(('image',name,caption,width))

b=page('Physics Judge interview preparation')
p(b,'Meta Research Scientist Level 5 contract\nRole context supplied by Kadir from Rahul’s briefing')
p(b,'Lead with the ability to define and test physical validity. The humanoid sitting project connects directly to contact, foot slip, support and environment interaction. It also provides an example of rejecting a result that looks plausible but fails the physical task. Direct generated-video evaluation and SMPL experience are the gaps to acknowledge.')
h(b,'Suggested opening answer')
p(b,'“My strongest connection is physics-based evaluation of humanoid motion. In our sitting project, a motion could look controlled yet fail because the feet slipped, the robot contacted the wrong surface, or it never established seated support. We separated those failure modes and evaluated complete trials. The current baseline is clean in 64 of 72 trials; the remaining eight are unresolved. That experience is relevant to defining what a physics judge should measure and testing whether optimizing its score improves behavior. My main gap is direct work with generated human video and SMPL. I would bring the contact-modeling and evaluation experience while building expertise in reconstructing human and object motion from video.”')
p(b,'Use this as a project-level draft. Replace “our” with an accurate account of what you personally designed, implemented or reviewed; AI coding assistance was part of this project.')
table(b,['Role objective','Connection to this project','Evidence boundary'],[
('Smarter judgments','Separate slip, illegal contact, missing support and task completion.','64/72 clean simulation trials; eight unresolved.'),
('Faster and cheaper training signals','Account for the whole pipeline; distinguish a useful objective from a score that can be exploited.','No GPU speedup or video-training result demonstrated.'),
('Human interaction with the scene','Model support surfaces, contact geometry and load transfer across time.','Sitting/contact experience; human grasping remains a new application.'),
],[1.55,2.75,2.5])
h(b,'How to frame the experience gap')
p(b,'“I have relevant simulation, PyTorch and contact-evaluation experience. I have not yet built a production judge for generated human video or demonstrated an SMPL pipeline. I understand that the central additional problem is uncertain 3D reconstruction: camera motion, scale and occlusion can look like physical violations. I would explicitly represent that uncertainty rather than treating every bad reconstruction as bad physics.”')
p(b,'Keep the 38% reduction in offline model mismatch as a secondary numerical-engineering example. It did not pass motion admission and is not a 38% gain in sitting or judge accuracy.')

b=page('One concrete example to tell')
h(b,'A hovering robot can receive a misleadingly good score')
p(b,'Situation: the goal was stable low-seat sitting without foot slip or unwanted housing contact. A candidate could hover near the seat without establishing the required support. A judge based only on low velocity, no slip or a plausible final pose could mistakenly reward it.')
image(b,'selected_recorded_trials.png','Saved simulation evidence. Ret09 succeeds at 36 cm / +5°; the other tiles show failures from separate experiments at 36 cm / +15°. Candidate 273 failed to reach stable support. These schematic poses come from recorded transforms; the labels reflect full-trial outcomes. The tiles are illustrative, not a matched comparison.',6.0)
p(b,'Action: evaluate the full 20-second attempt and keep the failure checks separate. In this robot study, the resting definition includes trunk tilt at most 40° and legal seat normal load at least 5% of body weight, alongside the other slip, contact and stability criteria. Candidate 273 lacked legal seat load and the required quiet release, so it was rejected.')
p(b,'Result: the evaluation prevented promotion of a false success. It did not repair the controller; the verified baseline remains 64/72. Earlier corrections also aligned a 51.6° planned endpoint with the existing 40° objective and corrected a roughly 5–6 mm planning/collision-model discrepancy.')
p(b,'Connection to Physics Judge: score whether the required interaction actually happens over time. In video, support must be inferred with uncertainty; the robot-specific load and angle thresholds cannot simply be copied into a human-video judge. The lesson is the evaluation structure, not those particular constants.')

b=page('How I would design the judge')
p(b,'Interview proposal, not an implemented system or a description of Meta’s internal architecture: reconstruct camera, human, object and scene motion; infer interaction phases and confidence; then produce localized physical-violation scores. Preserve the component scores and uncertainty even if training ultimately needs a scalar reward.')
table(b,['Failure','Candidate measurement','Important counterexample'],[
('Foot sliding','Tangential foot velocity relative to its supporting surface during an inferred stance interval.','Moving camera or platform; a swinging foot is not planted.'),
('Unsupported floating','Check inferred support contacts and whether motion is consistent with the expected supported or airborne phase.','Jumping, hanging or brief loss of contact can be valid.'),
('Failed grasp','Combine hand–object proximity, finger configuration, contact persistence and relative motion.','Co-motion alone does not prove a grasp or force closure.'),
('Interpenetration','Use an appropriate collision representation to measure penetration depth, duration and affected area.','Reconstruction error, clothing and soft contact can create apparent intersections.'),
('Broken interaction timing','Check approach, contact, transfer and release as a sequence; localize violations in time.','A plausible final frame can hide an impossible transition.'),
],[1.35,2.95,2.5])
h(b,'Separate missing evidence from a physical failure')
p(b,'Evaluate in a consistent world or surface frame and use physical time units. Carry confidence for pose, depth, scale and contact. An occluded hand should be marked uncertain when the evidence is insufficient. Do not silently treat uncertainty as success; report how much of the clip was actually assessed and test whether generators exploit blind spots.')
p(b,'A fitted body mesh is a geometric hypothesis. It does not establish contact forces, mass distribution or dynamically feasible motion. For ambiguous clips, consider multiple reconstructions or sensitivity bounds. If a simulator is used as a more expensive check, validate the mapping from estimated human motion to its body and contact model.')
h(b,'A bounded first experiment to propose')
p(b,'Start with one failure class, such as planted-foot sliding. Build paired clips with controlled slip perturbations and matched valid examples, including camera motion and moving supports. Mix simulated examples with human-labeled real and generated clips. Hold out subjects, scenes, camera setups and generator families where feasible. Freeze the benchmark before tuning thresholds.')
p(b,'Measure per-class precision and recall, severity ranking, timing localization and calibration at a stated coverage level. Compare the new component with a simpler baseline and ablate pose quality, camera compensation and contact inference. A higher average score alone would not demonstrate a better judge.')

b=page('Making the score useful for training')
h(b,'A practical path from evaluator to reward')
p(b,'A judge that scores completed videos can already provide a black-box terminal reward for reinforcement learning. Differentiability is not required for policy-gradient training; DDPO is a published example for diffusion models. Direct gradient guidance requires a differentiable path or suitable surrogate. [W4]')
bullets(b,[
    'First validate an offline judge against a fixed benchmark and human judgments. Keep an independent evaluator that is not optimized by the generator.',
    'Combine physical validity with task or prompt fulfillment. A motionless person, a hidden hand or an omitted object must not earn a high score simply by avoiding a detectable violation.',
    'Train a cheaper surrogate on expensive judge outputs if profiling justifies it. Audit disagreement and hard cases, including new outputs from the changing generator.',
    'For denoising-time guidance, validate inputs at the relevant noise levels or predicted clean reconstructions. A clean-video judge should not be assumed reliable on noisy intermediate states.',
    'Run a controlled training comparison at equal compute and prompts. Check physical quality, diversity, task completion and blind-spot exploitation with the independent evaluator.',
])
h(b,'How I would approach speed and cost')
p(b,'Profile decoding, pose and object estimation, body-model evaluation, geometric checks and data transfer separately. Batch suitable work across frames and clips, cache static topology, use coarse collision rejection before expensive tests, and allocate extra computation to uncertain contact windows. Test mixed precision only where it preserves sensitivity to small violations.')
p(b,'PyTorch’s profiler can expose both CPU and CUDA activity. Measure warmed-up end-to-end throughput, tail latency, peak memory and cost per clip on fixed hardware and video lengths; account correctly for asynchronous GPU execution. Any speed claim should hold at comparable detection quality and coverage. [W5]')
h(b,'What would count as success')
table(b,['Dimension','Evidence to seek'],[
('Judge quality','Better detection and calibration on held-out failure categories, not just correlation on an easy aggregate.'),
('Training utility','Generator improvement judged independently at equal training cost; no collapse into stillness or loss of prompt fulfillment.'),
('Efficiency','Lower cost or latency at matched quality, including reconstruction and transfer overhead.'),
('Expanded capability','Reliable results on new human–object and human–scene interactions, with uncertainty and coverage reported.'),
],[1.5,5.3])
p(b,'Project lesson to connect: a prior robot reward study produced 44/64 and 42/64 successes after training versus 47/64 for the starting policy. Neither was promoted. That separate experiment is a useful example of why optimizing a reward must be checked against actual outcomes.')

b=page('Body models and interview practice')
h(b,'SMPL concepts to know')
p(b,'SMPL represents human body shape and pose as a skinned 3D mesh. SMPL-H adds articulated hands through the MANO hand model; SMPL-X includes articulated hands and an expressive face. These representations are useful for tracking body surfaces and interactions. They are not, by themselves, physics simulators or proofs of physical plausibility. [W1–W3]')
p(b,'Be ready to discuss pose versus shape parameters, root motion, joint rotations, skinning, fitting to images, temporal consistency, camera coordinates and scale ambiguity. For grasp evaluation, hand articulation and object geometry matter. Explicitly distinguish understanding these concepts from hands-on SMPL implementation experience.')
h(b,'Questions to rehearse')
bullets(b,[
    'How would you distinguish foot sliding from camera motion, and how would you test the detector?',
    'How can a judge distinguish hovering from a jump or valid hanging support?',
    'What evidence supports a grasp when the fingers are occluded?',
    'How would you convert an expensive, nondifferentiable score into a training signal?',
    'What would the generator exploit, and how would you detect reward overoptimization?',
    'Which part of your project did you personally own, and what would you change about its research process?',
])
h(b,'Questions to ask the team')
p(b,'What does the current judge consume and return? Which failure classes are most costly? What are its measured latency and quality baselines? Is the intended integration offline labeling, RL post-training or denoising-time guidance? What outcome would define a successful first three months?')
h(b,'Prompt for ChatGPT')
p(b,'Prepare me for the Meta Research Scientist Level 5 Physics Judge interview described here. Use this brief and the project report as evidence. Ask what I personally implemented before writing first-person answers. Run a mock first round with a 90-second introduction, the hovering-versus-support example, and technical follow-ups on contact, uncertainty, rewards, PyTorch performance and SMPL. Challenge unsupported claims. Do not claim the eight sitting failures were repaired or that I have generated-video, SMPL or GPU-optimization results that are not documented.')
h(b,'Primary references')
for title,url in [
    ('W1  SMPL official project','https://smpl.is.tue.mpg.de/'),
    ('W2  MANO and SMPL-H official project','https://mano.is.tue.mpg.de/'),
    ('W3  SMPL-X official project','https://smpl-x.is.tue.mpg.de/'),
    ('W4  Training Diffusion Models with Reinforcement Learning','https://rl-diffusion.github.io/'),
    ('W5  PyTorch profiler documentation','https://docs.pytorch.org/docs/stable/profiler.html'),
]:b.append(('web',title,url))
p(b,'Project evidence: current_successor_checkpoint.json; the inherited low-seat protocol and handoff; all_attempts_video_307; and the separate reward-study REPORT.md, all documented in the companion project report. Role and terms are from Rahul’s briefing as supplied by Kadir: 12 months, remote, $160/hour W-2 without benefits; two rounds expected, first focused on experience and fit with possible technical questions.')

# Preserve the established document styling; do not execute the original builder.
source=(BASE/'build_report.py').read_text()
render=source[source.index('doc=Document()'):source.index('doc.core_properties.title=')]
render=render.replace("normal.font.size=Pt(10.5)","normal.font.size=Pt(11)")
render=render.replace("'HUMANOID SITTING CONTROL  |  PROJECT REPORT'","'PHYSICS JUDGE  |  INTERVIEW PREPARATION'")
render=render.replace("fp.add_run('October 22, 2026  •  ')","fp.add_run('Interview preparation  •  ')")
# Add working hyperlinks for the web sources, with full URLs in the text version.
needle="        elif kind=='source':"
handler="""        elif kind=='web':
            para=doc.add_paragraph();para.paragraph_format.space_after=Pt(3)
            link=OxmlElement('w:hyperlink')
            rid=doc.part.relate_to(block[2], 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)
            link.set(qn('r:id'),rid)
            run=OxmlElement('w:r');rp=OxmlElement('w:rPr')
            color=OxmlElement('w:color');color.set(qn('w:val'),'24578A');rp.append(color)
            size=OxmlElement('w:sz');size.set(qn('w:val'),'18');rp.append(size);run.append(rp)
            tx=OxmlElement('w:t');tx.text=block[1];run.append(tx);link.append(run);para._p.append(link)
            text += ['['+block[1]+']('+block[2]+')','']
"""
render=render.replace(needle,handler+needle)
OUT.mkdir(exist_ok=True)
exec(compile(render,'report_renderer','exec'))
doc.core_properties.title='Physics Judge interview preparation'
doc.core_properties.author='Prepared with Codex for Kadir'
doc.core_properties.subject='Role-specific framing, evidence and technical practice'
doc.save(OUT/'Physics_Judge_Interview_Brief.docx')
(OUT/'Physics_Judge_ChatGPT_Brief.txt').write_text('\n'.join(text))
(OUT/'content.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False))
files=[OUT/'Physics_Judge_Interview_Brief.docx',OUT/'Physics_Judge_ChatGPT_Brief.txt']
(OUT/'provenance.json').write_text(json.dumps({'role_source':'User-supplied Rahul briefing','scope':'Interview preparation only; no scientific execution','project_evidence':str(BASE/'provenance.json'),'web_sources':[x[2] for pg in pages for x in pg['blocks'] if x[0]=='web'],'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
print(json.dumps({'pages_designed':len(pages),'words':len(' '.join(text).split()),'outputs':[str(p) for p in files]}))
