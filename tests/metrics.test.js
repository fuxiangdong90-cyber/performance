import test from 'node:test';
import assert from 'node:assert/strict';
import {summarize, formulaValue, compareValues, csvCell} from '../web/metrics.js';
const row=(l,r,stage='forward')=>({paired:true,speedup:l/r,stage,execution:'eager',left:{wall_us:l,status:'pass'},right:{wall_us:r,status:'pass'}});
test('geomean, threshold and invalid/missing exclusions',()=>{
  const s=summarize([row(4,1),row(1,4),row(20,19),{paired:false,left:{status:'oom'},right:null}]);
  assert.equal(s.paired,3);assert.equal(s.failures,1);assert.equal(s.missing,1);
  assert.equal(s.left,1);assert.equal(s.right,1); // exactly 5% is not above 5%
  assert.ok(Math.abs(s.groups[0].value-Math.pow(20/19,1/3))<1e-12);
});
test('stages stay separate',()=>assert.equal(summarize([row(2,1),row(4,1,'backward')]).groups.length,2));
test('unknown stream and enqueue never become zero',()=>{
  assert.equal(formulaValue({right:{status:'pass',gpu_us:null,cpu_us:1}},'difference','right'),null);
  assert.equal(formulaValue({right:{status:'pass',gpu_us:10,cpu_us:4,wall_us:20}},'cpu_ratio','right'),.4);
});
test('null sorts last in both directions',()=>{assert.ok(compareValues(null,1,1)>0);assert.ok(compareValues(null,1,-1)>0);});
test('CSV prevents formula execution and preserves quotes',()=>{assert.equal(csvCell('=1+1'),'"\'=1+1"');assert.equal(csvCell('a"b'),'"a""b"');});
