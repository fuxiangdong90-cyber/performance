export function summarize(rows) {
  const paired = rows.filter(r => r.paired && Number.isFinite(r.speedup) && r.speedup > 0);
  const groups = {};
  for (const r of paired) {
    const key = `${r.stage} · ${r.execution}`;
    (groups[key] ||= []).push(r.speedup);
  }
  return {
    total: rows.length, paired: paired.length,
    right: paired.filter(r => r.right.wall_us < r.left.wall_us * .95).length,
    left: paired.filter(r => r.left.wall_us < r.right.wall_us * .95).length,
    failures: rows.filter(r => [r.left, r.right].some(x => x && x.status !== 'pass')).length,
    missing: rows.filter(r => !r.left || !r.right).length,
    groups: Object.entries(groups).map(([label, values]) => ({label, count: values.length, value: Math.exp(values.reduce((s, v) => s + Math.log(v), 0) / values.length)}))
  };
}

export function formulaValue(row, formula, side) {
  const v = row[side];
  if (!v || v.status !== 'pass' || v.gpu_us == null) return null;
  if (formula === 'difference') return v.cpu_us == null ? null : v.gpu_us - v.cpu_us;
  if (formula === 'cpu_ratio') return v.cpu_us == null || !v.gpu_us ? null : v.cpu_us / v.gpu_us;
  if (formula === 'gpu_ratio') return v.wall_us ? v.gpu_us / v.wall_us : null;
  return null;
}

export function valueAt(row, path) {
  return path.split('.').reduce((v, key) => v == null ? null : v[key], row);
}

export function compareValues(a, b, direction = 1) {
  if (a == null) return b == null ? 0 : 1;
  if (b == null) return -1;
  return direction * (typeof a === 'number' && typeof b === 'number' ? a - b : String(a).localeCompare(String(b), undefined, {numeric: true}));
}

export function csvCell(value) {
  let s = value == null ? '' : String(value);
  if (/^[=+\-@\t\r]/.test(s)) s = "'" + s;
  return '"' + s.replaceAll('"', '""') + '"';
}
