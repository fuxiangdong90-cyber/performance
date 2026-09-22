import json,tempfile,unittest
from pathlib import Path
from opbench.legacy_compare import compare
class LegacyCompareTests(unittest.TestCase):
 def test_pairs_exclusions_and_metric_separation(self):
  with tempfile.TemporaryDirectory() as tmp:
   a,b=Path(tmp)/'a',Path(tmp)/'b'
   def row(name,metric,value):return {'benchmark':{'mode':'inference','dtype':'float','extra_info':{'input_config':'M: 4','operator_name':'mm'}},'model':{'name':name},'metric':{'unit':'us','name':metric,'benchmark_values':[value]}}
   for root,rows in [(a,[row('mm','latency',10),row('mm','gpu stream latency',2),row('missing','latency',4),row('duplicate','latency',5),row('duplicate','latency',6)]),(b,[row('mm','latency',20),row('mm','gpu stream latency',1),row('duplicate','latency',7)])]:
    (root/'json').mkdir(parents=True);(root/'json'/'run.json').write_text(json.dumps(rows))
   d=compare(a,b);self.assertEqual(d['paired'],2);self.assertEqual(d['large_differences'],2);self.assertEqual(d['excluded_duplicate_keys']['baseline'],1);self.assertEqual(len(d['groups']),2)
