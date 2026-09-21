import {summarize, formulaValue, valueAt, compareValues, csvCell} from './metrics.js';
const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels = {matrix:'矩阵乘法',convolution:'卷积',transpose_convolution:'转置卷积',normalization:'归一化',activation:'激活函数',optimizer:'优化器'};
const stages = {forward:'前向',backward:'反向',optimizer:'参数更新'};
const state = {runs:[], rows:[], selected:new Set(), operators:[], page:1, size:50, sort:'name', direction:1, columnFilters:{}, pageName:'dashboard', compare:null, language:'zh'};
let requestId = 0;
let toastTimer;
const columns = [
  ['operator','算子','Operator'],['name','模型 / 形状','Model / shape'],['hardware','硬件','Hardware'],['dtype','数据类型','Dtype'],['stage','模式','Stage'],
  ['flops','单次计算量','FLOPs', 'flops'],['bytes','估算显存搬运量','Logical traffic','bytes'],['arithmetic_intensity','算存比','Intensity','intensity'],
  ['left.wall_us','左侧延迟','Left latency','time'],['right.wall_us','右侧延迟','Right latency','time'],
  ['left.gpu_us','左侧 GPU Stream 延迟','Left GPU stream','time'],['right.gpu_us','右侧 GPU Stream 延迟','Right GPU stream','time'],['gpu_speedup','GPU Stream 加速比','GPU speedup','ratio'],
  ['left.cpu_us','左侧 CPU 提交延迟','Left CPU enqueue','time'],['right.cpu_us','右侧 CPU 提交延迟','Right CPU enqueue','time'],
  ['left.tflops','左侧有效 TFLOPS','Left TFLOPS','number'],['right.tflops','右侧有效 TFLOPS','Right TFLOPS','number'],['speedup','加速比（左/右）','Speedup (L/R)','ratio'],
  ['left.bandwidth_gbs','左侧带宽','Left bandwidth','bandwidth'],['right.bandwidth_gbs','右侧带宽','Right bandwidth','bandwidth'],
  ['left.peak_allocated_bytes','左侧峰值内存','Left peak memory','bytes'],['right.peak_allocated_bytes','右侧峰值内存','Right peak memory','bytes']
];
function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 7000); }
async function api(path, body) {
  const headers = {};
  const token = sessionStorage.getItem('opbench-token');
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const r = await fetch(path, {headers, method:body === undefined ? 'GET':'POST', body:body === undefined ? undefined:JSON.stringify(body)});
  const data = await r.json();
  if (!r.ok) {
    if (r.status === 401 && !$('token-dialog').open) $('token-dialog').showModal();
    throw new Error(data.error || `HTTP ${r.status}`);
  }
  return data;
}
function fmt(v, type) {
  if (v == null) return '—';
  if (!type) return esc(v);
  if (type === 'time') return v >= 1000 ? `${(v/1000).toFixed(2)} ms` : `${v.toFixed(v < 10 ? 2 : 1)} μs`;
  if (type === 'ratio') return `${v.toFixed(2)}×`;
  if (type === 'bytes') {
    for (const [n,u] of [[1073741824,'GiB'],[1048576,'MiB'],[1024,'KiB']]) if (v >= n) return `${(v/n).toFixed(2)} ${u}`;
    return `${v.toFixed(0)} B`;
  }
  if (type === 'flops') return v >= 1e9 ? `${(v/1e9).toFixed(2)} GFLOP` : `${(v/1e6).toFixed(2)} MFLOP`;
  if (type === 'intensity') return `${v.toFixed(2)} FLOP/B`;
  if (type === 'bandwidth') return `${v.toLocaleString(undefined,{maximumFractionDigits:2})} GB/s`;
  return v.toFixed(2);
}
function header() {
  $('results').querySelector('thead').innerHTML = `<tr>${columns.map(([key,zh,en,type]) => `<th><button data-sort="${key}">${esc(state.language === 'zh' ? zh : en)} ${state.sort === key ? (state.direction === 1 ? '↑':'↓') : '↕'}</button><input data-filter="${key}" aria-label="筛选 ${esc(zh)}" placeholder="${type ? '≥ 最小值 / ≤ 最大值':'包含…'}" value="${esc(state.columnFilters[key] || '')}"></th>`).join('')}</tr>`;
  $('results').querySelectorAll('[data-sort]').forEach(b => b.onclick = () => {state.direction = state.sort === b.dataset.sort ? -state.direction : 1; state.sort = b.dataset.sort; header(); render();});
  $('results').querySelectorAll('[data-filter]').forEach(input => input.oninput = () => {state.columnFilters[input.dataset.filter] = input.value; state.page = 1; render();});
}
function setOptions(id, values, first) {
  const previous = $(id).value;
  $(id).innerHTML = `<option value="">${first}</option>` + values.map(v => `<option value="${esc(v)}">${esc(labels[v] || v)}</option>`).join('');
  if (values.includes(previous)) $(id).value = previous;
}
async function loadRuns() {
  const runs = await api('/api/runs');
  state.runs = runs;
  const l = $('left').value, r = $('right').value;
  const options = runs.map(run => `<option value="${run.id}">${esc(run.name)} · ${esc(run.metadata.device.name)}</option>`).join('');
  $('left').innerHTML = $('right').innerHTML = options;
  if (runs.some(x => x.id === l)) $('left').value = l;
  else if (runs.length > 1) $('left').selectedIndex = 1;
  if (runs.some(x => x.id === r)) $('right').value = r;
  $('run-count').textContent = `${runs.length} 次运行可用`;
  $('data-count').textContent = `${runs.reduce((n,x) => n+x.count,0).toLocaleString()} 条结果 · ${runs.length} 次运行`;
  await loadComparison();
}
async function loadComparison() {
  const current = ++requestId;
  try {
    if (!$('left').value || !$('right').value) {
      state.rows = []; state.compare = null; state.operators = []; state.selected.clear();
      $('warnings').textContent = '尚无运行数据。导入测试脚本生成的 JSON，或使用 --demo 启动以查看合成演示。';
      $('comparison-label').textContent = ''; renderOperators(); render(); return;
    }
    const data = await api(`/api/compare?left=${encodeURIComponent($('left').value)}&right=${encodeURIComponent($('right').value)}`);
    if (current !== requestId) return;
    state.compare = data;
    state.rows = data.rows.map(r => ({...r, hardware:data.right.device.name}));
    state.operators = [...new Set(state.rows.map(r=>r.operator))].sort();
    state.selected = new Set(state.operators);
    state.page = 1;
    const notes = ['口径：加速比 = 左侧同步延迟 ÷ 右侧同步延迟；只汇总双方成功的同配置用例。反向访存未建模，带宽留空。'];
    const composite = [...new Set(data.rows.filter(r=>[r.left,r.right].some(x=>x?.implementation && x.implementation!=='native')).map(r=>r.operator))];
    if(composite.length) notes.push(`GPU 组合实现：${esc(composite.join('、'))}（实现方式见用例名称），与原生融合实现分开匹配。`);
    const checked = ['left','right'].map(side=>data.rows.filter(r=>r[side]?.correctness==='cpu_reference' && r[side]?.reference).length);
    if(checked.some(Boolean)) notes.push(`CPU 原生参考校验：左侧 ${checked[0]} 个，右侧 ${checked[1]} 个配置通过预设容差检查；误差与容差见用例详情。`);
    const rejected = ['left','right'].map(side=>data.rows.filter(r=>r[side]?.correctness==='cpu_reference_failed').length);
    if(rejected.some(Boolean)) notes.push(`CPU 参考校验超差：左侧 ${rejected[0]} 个，右侧 ${rejected[1]} 个；这些配置不参与性能汇总。`);
    $('warnings').innerHTML = [...notes,...data.warnings.map(esc)].join('<br>');
    $('comparison-label').textContent = `${data.left.name} → ${data.right.name}`;
    setOptions('dtype',[...new Set(state.rows.map(r=>r.dtype))].sort(),'全部数据类型');
    setOptions('category',[...new Set(state.rows.map(r=>r.category))],'全部分类');
    setOptions('hardware',[...new Set([data.left.device.name,data.right.device.name])],'全部硬件');
    renderOperators(); render();
  } catch (error) { if (current === requestId) {state.rows=[]; render(); $('warnings').textContent = error.message; toast(error.message);} }
}
function renderOperators() {
  const counts = Object.fromEntries(state.operators.map(op => [op,state.rows.filter(r=>r.operator===op).length]));
  const all = state.selected.size === state.operators.length;
  $('selection-count').textContent = all ? '全部已选' : `${state.selected.size} / ${state.operators.length} 项已选`;
  $('operator-list').innerHTML = `<button class="op ${all?'selected':''}" data-op="all">全部算子<small>${state.rows.length} 个用例</small></button>` + state.operators.map(op=>`<button class="op ${state.selected.has(op)?'selected':''}" data-op="${esc(op)}">${esc(op)}<small>${counts[op]} 个用例</small></button>`).join('');
  $('operator-list').querySelectorAll('button').forEach(b => {b.setAttribute('aria-pressed', b.classList.contains('selected')); b.onclick=()=>{
    if (b.dataset.op === 'all') state.selected = new Set(all?[]:state.operators);
    else if (state.selected.has(b.dataset.op)) state.selected.delete(b.dataset.op); else state.selected.add(b.dataset.op);
    state.page=1; renderOperators(); render();
  };});
}
function filteredRows() {
  const query = $('search').value.toLowerCase();
  const formula = $('formula').value;
  const minimum = $('min').value === '' ? -Infinity : Number($('min').value);
  const maximum = $('max').value === '' ? Infinity : Number($('max').value);
  return state.rows.filter(r => {
    if (!state.selected.has(r.operator) || !`${r.name} ${r.operator} ${JSON.stringify(r.params)}`.toLowerCase().includes(query)) return false;
    for (const key of ['category','dtype','stage']) if ($(key).value && r[key] !== $(key).value) return false;
    if ($('hardware').value && ![state.compare.left.device.name,state.compare.right.device.name].includes($('hardware').value)) return false;
    const s = $('status').value;
    if (s === 'paired' && !r.paired || s === 'failure' && ![r.left,r.right].some(x=>x && x.status !== 'pass') || s === 'missing' && r.left && r.right) return false;
    if (formula) {const value = formulaValue(r,formula,$('formulaSide').value); if (value == null || value < minimum || value > maximum) return false;}
    for (const [key,text] of Object.entries(state.columnFilters)) {
      if (!text.trim()) continue;
      const value = valueAt(r,key);
      if (value == null) return false;
      const match = text.trim().match(/^(>=|<=|>|<|=|≥|≤)\s*(-?\d+(?:\.\d+)?)$/);
      if (match) {
        const n=Number(match[2]); const v=Number(value);
        if (!Number.isFinite(v)) return false;
        if (!({'>' : v>n,'<' : v<n,'=' : v===n,'>=':v>=n,'<=':v<=n,'≥':v>=n,'≤':v<=n}[match[1]])) return false;
      } else if (!String(value).toLowerCase().includes(text.toLowerCase())) return false;
    }
    return true;
  }).sort((a,b)=>compareValues(valueAt(a,state.sort),valueAt(b,state.sort),state.direction));
}
function render() {
  const rows = filteredRows(); const summary = summarize(rows);
  $('summary').innerHTML = `<article class="stat"><label>参与汇总用例</label><strong>${summary.paired.toLocaleString()}</strong><small>${summary.total.toLocaleString()} 个可见配置 · ${summary.missing} 个缺失匹配</small></article><article class="stat"><label>几何平均加速比</label>${summary.groups.length?summary.groups.map(g=>`<small>${esc(g.label)} · n=${g.count}</small><strong class="geomean ${g.value>=1?'green':'blue'}">${g.value.toFixed(2)}×</strong>`).join(''):'<strong>—</strong>'}<small>左侧延迟 ÷ 右侧延迟</small></article><article class="stat"><label>右侧更快</label><strong class="green">${summary.right}</strong><small>右侧延迟低 5% 以上</small></article><article class="stat"><label>左侧更快</label><strong class="blue">${summary.left}</strong><small>左侧延迟低 5% 以上</small></article><button class="stat" id="show-failures"><label>失败 / 不支持</label><strong class="${summary.failures?'red':''}">${summary.failures}</strong><small>点击查看失败详情</small></button>`;
  $('show-failures').onclick=()=>{$('status').value='failure';state.page=1;render();};
  const pages = Math.max(1,Math.ceil(rows.length/state.size)); state.page=Math.min(pages,state.page);
  const visible=rows.slice((state.page-1)*state.size,state.page*state.size);
  $('results').querySelector('tbody').innerHTML = visible.length ? visible.map(r=>`<tr>${columns.map(([key,,,type])=>{
    const value=valueAt(r,key); let html=fmt(value,type); let cls='';
    if(key==='operator') html=`<button class="link" data-isolate="${esc(r.operator)}">${esc(r.operator)}</button>`;
    if(key==='name') {const status = !r.left || !r.right ? 'missing' : [r.left,r.right].find(x=>x.status!=='pass')?.status || 'pass'; html=`<button class="link" data-detail="${r.case_key}">${esc(r.name)}</button><span class="badge ${status}">${status.toUpperCase()}</span>`;cls='name';}
    if(key==='stage') html=`${esc(stages[r.stage])}<small class="muted"> · ${esc(r.module_mode)}</small>`;
    if(key==='hardware') html=esc(state.compare.right.device.name);
    if(key==='speedup' || key==='gpu_speedup') cls=value==null?'muted':value>1/.95?'green':value<.95?'blue':'';
    if(key.includes('.') && ['failed','unsupported','oom'].includes(r[key.split('.')[0]]?.status)) html='—';
    return `<td class="${cls}" title="${esc(key==='name'?r.name:html.replace(/<[^>]*>/g,''))}">${html}</td>`;
  }).join('')}</tr>`).join('') : `<tr><td colspan="${columns.length}" class="empty">没有匹配的基准数据。请导入 JSON 文件或调整筛选条件。</td></tr>`;
  $('results').querySelectorAll('[data-isolate]').forEach(b=>b.onclick=()=>{state.selected=new Set([b.dataset.isolate]);state.page=1;renderOperators();render();});
  $('results').querySelectorAll('[data-detail]').forEach(b=>b.onclick=()=>showDetails(b.dataset.detail));
  $('page-label').textContent=`显示 ${rows.length?(state.page-1)*state.size+1:0} – ${Math.min(state.page*state.size,rows.length)} 共 ${rows.length}`;
  $('page-number').textContent=`第 ${state.page} / ${pages} 页`;
  $('previous').disabled=state.page===1; $('next').disabled=state.page===pages;
}
function showDetails(key) {
  const r=state.rows.find(r=>r.case_key===key);
  $('detail-content').innerHTML=`<p>${esc(r.name)}</p><p>${esc(r.workload_note)}</p><div class="grid">${['left','right'].map(side=>`<section><h2>${side==='left'?'左侧':'右侧'}运行</h2><pre>${esc(JSON.stringify(r[side],null,2))}</pre></section>`).join('')}</div>`;
  $('details').showModal();
}
function download(name,text,type='application/json') {
  const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function importFiles(files) {
  let count=0;
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith('.json')) continue;
    try {if(file.size>32*1024*1024) throw new Error('文件超过 32 MiB'); const result=await api('/api/import',JSON.parse(await file.text())); count++; toast(`${file.name}：已导入 ${result.count} 个用例`);}
    catch(e){toast(`${file.name}：${e.message}`);break;}
  }
  if(count) {await loadRuns();if(state.pageName==='runs') await navigate('runs');}
}
async function navigate(page) {
  state.pageName=page;
  document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.dataset.page===page));
  $('dashboard').hidden=page!=='dashboard'; $('secondary').hidden=page==='dashboard';
  if(page==='dashboard') return;
  $('secondary').innerHTML='<p>正在加载…</p>';
  try {
    if(page==='runs') await renderRuns(false);
    if(page==='devices') {
      const devices=await api('/api/devices');
      $('secondary').innerHTML=`<div class="secondary-hero"><div class="eyebrow">HARDWARE INVENTORY</div><h1>设备</h1><p>由导入报告中的设备元数据自动建立。</p></div><div class="grid">${devices.map(d=>`<article class="panel card"><h2>${esc(d.name)}</h2><p>${esc(d.backend)}</p><pre>${esc(JSON.stringify(d,null,2))}</pre></article>`).join('') || '<p>暂无设备记录。</p>'}</div>`;
    }
    if(page==='templates') {
      const templates=await api('/api/templates');
      $('secondary').innerHTML=`<div class="secondary-hero"><div class="eyebrow">REPRODUCIBLE BENCHMARKS</div><h1>测试模板</h1><p>声明式配置，不执行模板内的任意代码。所有模板均可下载、展开校验。</p></div><div class="grid">${templates.map(t=>`<article class="panel card"><h2>${esc(t.name)}</h2><p>${esc(t.description)}</p><p>${t.operators.length} 种算子</p><pre>python -m opbench.runner --template templates/${esc(t.id)}.json --device cpu --output results/run.json</pre><div class="actions"><button data-download="${esc(t.id)}" class="primary">下载模板</button><button data-validate="${esc(t.id)}">展开校验</button></div><pre id="validation-${esc(t.id)}" hidden></pre></article>`).join('')}</div>`;
      $('secondary').querySelectorAll('[data-download]').forEach(b=>b.onclick=()=>{const {id,...t}=templates.find(t=>t.id===b.dataset.download);download(`${id}.json`,JSON.stringify(t,null,2));});
      $('secondary').querySelectorAll('[data-validate]').forEach(b=>b.onclick=async()=>{try {const t=templates.find(t=>t.id===b.dataset.validate);const v=await api('/api/templates/validate',t);const el=$(`validation-${t.id}`);el.hidden=false;el.textContent=`校验成功：${v.count} 个唯一用例\n`+JSON.stringify(v.cases[0],null,2);} catch(e){toast(e.message);}});
    }
    if(page==='docs') $('secondary').innerHTML=docs;
  } catch(e){$('secondary').innerHTML=`<div class="error-box">${esc(e.message)}</div>`;}
}
async function renderRuns(archived) {
  const runs=await api(`/api/runs?archived=${archived?1:0}`);
  $('secondary').innerHTML=`<div class="secondary-hero"><div class="eyebrow">EXPERIMENT HISTORY</div><h1>运行记录</h1><p>保存完整环境、样本与失败信息。归档可恢复。</p></div><div class="view-toolbar"><button id="active-runs" ${!archived?'class="primary"':''}>当前运行</button><button id="archived-runs" ${archived?'class="primary"':''}>已归档</button></div><div class="grid">${runs.map(r=>`<article class="panel card"><h2>${esc(r.name)} ${r.synthetic?'<span class="badge unsupported">SYNTHETIC</span>':''}</h2><p>${esc(r.metadata.device.name)}</p><p>${r.count} 个用例 · ${r.failures} 个失败 / 不支持</p><p>${esc(new Date(r.created_at).toLocaleString())}</p><details><summary>运行环境</summary><pre>${esc(JSON.stringify(r.metadata,null,2))}</pre></details><div class="actions"><button data-export-run="${r.id}">导出 JSON</button><button data-archive="${r.id}">${archived?'恢复运行':'归档运行'}</button></div></article>`).join('') || '<p>暂无记录。</p>'}</div>`;
  $('active-runs').onclick=()=>renderRuns(false).catch(e=>toast(e.message));$('archived-runs').onclick=()=>renderRuns(true).catch(e=>toast(e.message));
  $('secondary').querySelectorAll('[data-export-run]').forEach(b=>b.onclick=async()=>{try{download(`run-${b.dataset.exportRun}.json`,JSON.stringify(await api(`/api/runs/${b.dataset.exportRun}`),null,2));}catch(e){toast(e.message);}});
  $('secondary').querySelectorAll('[data-archive]').forEach(b=>b.onclick=async()=>{try{await api(`/api/runs/${b.dataset.archive}/archive`,{archived:!archived});await loadRuns();await renderRuns(archived);}catch(e){toast(e.message);}});
}
const docs=`<div class="secondary-hero"><div class="eyebrow">METRICS & METHODOLOGY</div><h1>使用文档</h1><p>从采集到分析，保持每个数字的含义明确。</p></div><article class="panel doc"><h2>1. 采集与导入</h2><pre>python -m opbench.runner --template templates/smoke.json --device cpu --output results/cpu.json\npython -m opbench.runner --template templates/standard.json --device cuda --warmup 10 --iterations 50 --output results/gpu.json\npython -m opbench.runner --template templates/smoke.json --device cpu --upload http://127.0.0.1:30000</pre><p>脚本需要单独安装适合目标设备的 PyTorch。服务端和网页不依赖 PyTorch。CPU 测试的 GPU Stream、CPU 提交和显存指标为 null，不伪造 GPU 数据。</p><h2>2. 指标定义</h2><p>主延迟为逐次同步的 wall time 中位数，包含主机提交与结束同步。有效 TFLOPS = FLOPs / wall_seconds / 10¹²；有效带宽 = logical_bytes / wall_seconds / 10⁹。GPU Stream 为事件区间，可能包含流空闲；CPU 提交不含末尾同步。三者不能简单相加减作为纯 kernel 开销。</p><p>流量是算法逻辑估算，不是实测 HBM 流量。矩阵乘法 FLOPs 只统计主体乘加，FMA=2；卷积使用直接卷积等效量。未建模的计算量/反向访存显示 —。峰值内存为框架 allocated bytes；CPU 不提供该指标。</p><h2>3. 前向 / 反向 / 更新</h2><p>forward 使用 no_grad；backward 在计时外构造计算图，只测所有可求导输入和参数的 autograd.grad；optimizer 测已初始化状态的单次 step，每次测量前恢复参数和优化器状态。train/eval 是独立模块状态，不等同于前向/反向。</p><h2>4. 匹配与汇总</h2><p>用例按 operator + params + dtype + stage + execution + module_mode + gradient_scope + implementation 匹配。几何平均只纳入双方成功用例，按阶段和执行方式分组。双方失败、缺失和不支持单独显示。5% 是展示阈值，不是统计显著性检验。不同运行的环境差异会显示提示。</p><h2>5. 筛选与导出</h2><p>所有卡片统计当前筛选集，不只统计当前页。列过滤支持包含文本及 &gt;、&gt;=、&lt;、&lt;=、= 数值比较；数值采用原始单位：μs、B、FLOP、GB/s。算式过滤可选择左右侧。CSV 导出全部筛选行。</p><h2>6. 正确性和兼容性</h2><p>默认 finite_only 仅检查结果有限性。使用 --verify-reference 时，CPU float64 原生算子与目标设备使用相同量化输入校验输出或梯度，保留目标 dtype 的 RMSNorm epsilon；报告 cpu_reference 或 cpu_reference_failed、容差和误差；校验在计时外完成。addbmm 的 GPU 组合实现显式标记，并与 native 实现分开匹配。历史站点的 JSON 原始 schema 尚未取得；本系统仅导入 docs/report-format.md 定义的 v1 格式，拒绝猜测时间单位或字段语义。</p><h2>7. 服务与数据库</h2><p>SQLite WAL 存储设备、运行、用例与测量；默认只监听本机。远程部署设置 OPBENCH_API_TOKEN 并使用 HTTPS 反向代理。HTTP 不提供远程脚本执行；测试在目标机器运行，再上传 JSON。归档不会删除结果，可在运行记录中恢复。</p></article>`;
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>navigate(b.dataset.page));
$('left').onchange=$('right').onchange=loadComparison;
$('swap').onclick=()=>{const l=$('left').value;$('left').value=$('right').value;$('right').value=l;loadComparison();};
for(const id of ['search','category','hardware','dtype','stage','status','formula','formulaSide','min','max']) $(id).addEventListener(id==='search'||id==='min'||id==='max'?'input':'change',()=>{state.page=1;render();});
$('reset').onclick=()=>{for(const id of ['search','category','hardware','dtype','stage','status','formula','min','max']) $(id).value='';state.columnFilters={};state.selected=new Set(state.operators);state.page=1;header();renderOperators();render();};
$('previous').onclick=()=>{state.page--;render();};$('next').onclick=()=>{state.page++;render();};$('pageSize').onchange=()=>{state.size=Number($('pageSize').value);state.page=1;render();};
$('import').onclick=()=>$('upload').click();
for(const id of ['upload','folder']) $(id).onchange=async e=>{await importFiles(e.target.files);e.target.value='';};
$('close-details').onclick=()=>$('details').close();
$('export').onclick=()=>{const rows=filteredRows();const content=[columns.map(c=>csvCell(c[1]+' ['+c[0]+']')).join(','),...rows.map(r=>columns.map(([key])=>csvCell(valueAt(r,key))).join(','))].join('\r\n');download('opbench-comparison.csv','\ufeff'+content,'text/csv;charset=utf-8');};
$('language').onclick=()=>{state.language=state.language==='zh'?'en':'zh';$('language').textContent=state.language==='zh'?'EN':'中';document.documentElement.lang=state.language==='zh'?'zh-CN':'en';header();toast(state.language==='en'?'Table headings switched to English. 中文说明保留。':'表头已切换为中文');};
$('auth').onclick=()=>$('token-dialog').showModal();$('cancel-token').onclick=()=>$('token-dialog').close();
$('save-token').onclick=()=>{sessionStorage.setItem('opbench-token',$('token-input').value);$('token-input').value='';$('token-dialog').close();loadRuns().catch(e=>toast(e.message));};
header();loadRuns().catch(e=>{toast(e.message);$('data-count').textContent='服务连接失败';});
