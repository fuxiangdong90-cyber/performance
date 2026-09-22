"""Compare original HUD records without relabelling their timing semantics."""
import argparse,csv,json,math,statistics
from collections import defaultdict
from pathlib import Path

def read_records(root):
    files=sorted(Path(root).glob('json/*.json'))
    rows={}; duplicates=[]
    for path in files:
        data=json.loads(path.read_text())
        if not isinstance(data,list):continue
        for r in data:
            b,m,v=r.get('benchmark',{}),r.get('model',{}),r.get('metric',{})
            if v.get('unit')!='us' or v.get('name') not in ('latency','gpu stream latency','cpu enqueue latency'):continue
            info=b.get('extra_info',{})
            key=json.dumps([m.get('name'),b.get('mode'),b.get('dtype'),info.get('input_config'),info.get('use_compile',False),v.get('name')],ensure_ascii=False)
            values=v.get('benchmark_values',[])
            if not values or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in values):continue
            if key in rows:duplicates.append(key);continue
            rows[key]={'model':m.get('name'),'operator':info.get('operator_name',m.get('extra_info',{}).get('operator_name')),'stage':b.get('mode'),'dtype':b.get('dtype'),'metric':v['name'],'us':statistics.median(values),'source':str(path)}
    for key in duplicates:rows.pop(key,None)
    return rows,len(set(duplicates))

def compare(left,right,threshold=.2):
    a,ad=read_records(left);b,bd=read_records(right);rows=[];groups=defaultdict(list)
    for k in sorted(a.keys()|b.keys()):
        l,r=a.get(k),b.get(k);base=r or l
        ratio=l['us']/r['us'] if l and r else None
        change=r['us']/l['us']-1 if l and r else None
        row={**{x:base[x] for x in ('model','operator','stage','dtype','metric')},'baseline_us':l['us'] if l else None,'retest_us':r['us'] if r else None,'speedup':ratio,'relative_change':change,'large_difference':abs(change)>threshold if change is not None else False,'status':'paired' if l and r else 'missing_retest' if l else 'missing_baseline'}
        rows.append(row)
        if ratio is not None:groups[(base['stage'],base['metric'])].append(ratio)
    return {'threshold':threshold,'baseline':str(left),'retest':str(right),'excluded_duplicate_keys':{'baseline':ad,'retest':bd},'paired':sum(r['status']=='paired' for r in rows),'missing':sum(r['status']!='paired' for r in rows),'large_differences':sum(r['large_difference'] for r in rows),'groups':[{'stage':s,'metric':m,'pairs':len(v),'geomean_speedup':math.exp(sum(map(math.log,v))/len(v))} for (s,m),v in groups.items()],'rows':rows}

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--retest',required=True);p.add_argument('--output',required=True);p.add_argument('--threshold',type=float,default=.2);a=p.parse_args()
    if not 0<a.threshold<1:p.error('threshold must be between 0 and 1')
    d=compare(a.baseline,a.retest,a.threshold);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
    if d['rows']:
        with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(d['rows'][0]));w.writeheader();w.writerows(d['rows'])
    lines=['# Historical HUD comparison','',f"Paired metrics: {d['paired']}; missing metrics: {d['missing']}; changes over {a.threshold:.0%}: {d['large_differences']}.",'','Ratios are baseline / retest. Missing/duplicate cases are excluded. Metrics and stages stay separate. A difference is a diagnostic flag, not proof of optimization or a correctness failure.','']
    lines += [f"- {g['stage']} / {g['metric']}: {g['geomean_speedup']:.4f} ({g['pairs']} pairs)" for g in d['groups']]
    lines+=['','## Follow-up','', 'Review case coverage and status files first. For large changes, compare runtime versions, NUMA affinity, concurrent jobs, warmup/adaptive iteration policy and frequency logs. Repeat flagged workloads before changing the timing implementation. Do not loosen correctness thresholds to improve pass counts.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in d.items() if k!='rows'}))
if __name__=='__main__':main()
