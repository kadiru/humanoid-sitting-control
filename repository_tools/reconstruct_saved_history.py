"""Reconstruct an additive archive from saved versions, never invent past edits.

Uses existing Git blobs and a private index. Does not check out files, rewrite
main, backdate commits, or execute any historical code. Cross-family ordering
is only an archive order; same-day edit order is generally unavailable.
"""
from pathlib import Path
import argparse, collections, datetime, json, os, re, subprocess, tempfile
ROOT=Path(__file__).resolve().parents[1]
SOURCE='f7d47eddbda56b53b44aa82496456ebb4609d86f'
BRANCH='reconstructed-saved-history'

def git(*args, data=None, env=None):
    return subprocess.check_output(['git',*args],cwd=ROOT,input=data,env=env)

def natural(s):
    return tuple((0,int(x)) if x.isdigit() else (1,x) for x in re.split(r'(\d+)',s))

def grouping(path):
    bits=path.split('/')
    if path.startswith('ubuntu/experiments/'):
        family=bits[2];tail=bits[3:]
    else:family=bits[0];tail=bits[1:]
    if family in ('.gitattributes','.gitignore','README.md','ARTIFACTS.md','repository_metadata','repository_tools'):
        return 'repository-import-metadata'
    if family=='ubuntu':return 'ubuntu-runtime-snapshot'
    if re.search(r'2026[-]?\d{2}[-]?\d{2}',family):return family
    # The local numbered evidence bundles identify saved checkpoints, not edits.
    if tail:
        match=re.search(r'_(\d{1,3})(?:\.[^.]+)?$',tail[0])
        if match:return family+'/saved-record-'+str(int(match[1])).zfill(3)
    return family+'/unsequenced-records'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',action='store_true');a=parser.parse_args()
    source_tree=git('rev-parse',SOURCE+'^{tree}').decode().strip()
    entries=git('ls-tree','-rz',SOURCE).split(b'\0');groups=collections.defaultdict(list)
    for entry in entries:
        if not entry:continue
        header,path=entry.split(b'\t',1);mode,kind,sha=header.split()
        assert kind==b'blob'
        groups[grouping(path.decode())].append((mode,sha,path))
    keys=sorted(groups,key=natural)
    keys.remove('repository-import-metadata');keys.insert(0,'repository-import-metadata')
    manifest=dict(source_commit=SOURCE,source_tree=source_tree,branch=BRANCH,
        scope='All tracked files in the initial import; omitted bulk artifacts remain externally indexed.',
        limitations=['These are reconstructed archive commits created now, not recovered original Git commits.',
          'Archive order follows saved folder labels and local numeric record IDs, not verified global edit chronology.',
          'A record may contain a design, failure, audit or duplicate snapshot; it is not necessarily a code fix.',
          'Overwritten unsaved intermediate edits, original authors and exact edit times cannot be recovered.',
          'No historical program is executed and no training or simulation is restarted.'],
        groups=[dict(group=k,files=len(groups[k])) for k in keys],file_count=sum(map(len,groups.values())))
    target=ROOT/'repository_metadata/reconstructed_history.json'
    if a.build:
        exists=subprocess.run(['git','show-ref','--verify','--quiet','refs/heads/'+BRANCH],cwd=ROOT).returncode==0
        if exists:raise RuntimeError('Ref already exists; do not overwrite historical reconstruction')
        fd,index=tempfile.mkstemp(prefix='humanoid-history-index-');os.close(fd);os.unlink(index)
        env=dict(os.environ,GIT_INDEX_FILE=index);parent=None;records=[]
        try:
            git('read-tree','--empty',env=env)
            for i,k in enumerate(keys,1):
                records_for_index=b''.join(mode+b' '+sha+b'\t'+path+b'\0' for mode,sha,path in groups[k])
                git('update-index','-z','--index-info',data=records_for_index,env=env)
                tree=git('write-tree',env=env).decode().strip()
                args=['commit-tree',tree]+(['-p',parent] if parent else [])
                message=('Reconstruct saved records: '+k+'\n\nImported from '+SOURCE+'.\n'
                  'Reconstructed archive, not an original development commit.\n'
                  'Saved labels establish this grouping; exact historical edit order is not claimed.\n')
                parent=git(*args,data=message.encode()).decode().strip()
                records.append(dict(group=k,commit=parent,files=len(groups[k])))
                if i%100==0:print('reconstructed',i,'of',len(keys),flush=True)
            assert tree==source_tree,('TREE_MISMATCH',tree,source_tree)
            git('update-ref','refs/heads/'+BRANCH,parent,'0'*40)
            manifest.update(commits=records,commit_count=len(records),head=parent,final_tree_matches_source=True,
              created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        finally:
            if os.path.exists(index):os.unlink(index)
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('branch','file_count')},sort_keys=True),len(keys),'groups')
if __name__=='__main__':main()
