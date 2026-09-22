"""Durable historical HUD replay. Run on the benchmark host via systemd."""
import argparse,datetime,hashlib,json,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--container',required=True);p.add_argument('--job',required=True);a=p.parse_args()
 root=Path(a.root);job=Path(a.job);job.mkdir(parents=True,exist_ok=True)
 source=root/'repo';baseline=root/'results/bw1100-c8bbc83-pytorch-opbench-formal-r01'
 manifest=json.loads((baseline/'run-manifest.json').read_text());expected=manifest['environment']['image_digest']
 actual=subprocess.check_output(['docker','inspect',a.container,'--format','{{.Image}}'],text=True).strip()
 if actual!=expected:raise RuntimeError('Image differs from historical baseline')
 commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
 if commit!=manifest['source']['commit']:raise RuntimeError('Source commit differs from historical baseline')
 runner=source/'scripts/run-pytorch-operator-benchmark-hygon.sh'
 state={'phase':'prepared','source_commit':commit,'runner_sha256':hashlib.sha256(runner.read_bytes()).hexdigest(),'image_id':actual,'runs':[],'scope':'49122 baseline replay only; other hosts deferred by user','database_upload':'pending: direct route to backend unavailable','large_difference_threshold':.2}
 def save():
  state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(job/'status.json')
 save()
 names=[]
 for index in (1,2):
  name=job.name+'-r'+str(index);names.append(name);target=root/'results'/name
  if target.exists():raise RuntimeError('Refusing to overwrite run '+name)
  state['phase']='running_'+str(index);save()
  env={'RUN_NAME':name,'WORK_ROOT':'/workspace/opbench-hud/work/'+name,'RESULT_ROOT':'/workspace/results/'+name,'STATE_ROOT':'/tmp/'+name,'IMAGE_DIGEST':actual,'SMOKE_RESULT_ROOT':'/workspace/results/bw1100-c8bbc83-formal-smoke-r01','GPU_INDICES':'0,1,2,3,4,5,6,7','CPU_THREADS':'1','CORES_PER_DEVICE':'2','CASE_TIMEOUT_SECONDS':'7200'}
  command=['docker','exec','-w','/workspace/opbench-hud']
  for k,v in env.items():command+=['-e',k+'='+v]
  command += [a.container,'bash','/workspace/opbench-hud/scripts/run-pytorch-operator-benchmark-hygon.sh']
  with (job/(name+'.log')).open('w') as log:rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
  state['runs'].append({'name':name,'exit_code':rc,'result_root':str(target)});save()
  if not list((target/'json').glob('*.json')):state['phase']='blocked_no_result_records';save();return
  subprocess.run([sys.executable,str(job/'legacy_compare.py'),'--baseline',str(baseline),'--retest',str(target),'--output',str(job/('historical-comparison-r'+str(index)))],check=True)
 subprocess.run([sys.executable,str(job/'legacy_compare.py'),'--baseline',str(root/'results'/names[0]),'--retest',str(root/'results'/names[1]),'--output',str(job/'repeat-comparison')],check=True)
 d=json.loads((job/'repeat-comparison/comparison.json').read_text())
 state['phase']='retests_complete_review_required';state['repeat_large_differences']=d['large_differences'];state['next_step']='Review coverage, outliers, environment and repeat stability before freezing scripts or testing other hardware';save()
if __name__=='__main__':main()
