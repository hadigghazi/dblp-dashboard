import { useEffect, useRef, useState } from "react";
import * as echarts from "echarts";

// ---------------------------------------------------------------- theme ----
// ECharts draws to a canvas, which never resolves CSS custom properties.
// Every colour must be read off the document as a literal before it is used.
export function tok(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function useThemeVersion() {
  const [v, setV] = useState(0);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const bump = () => setV((x) => x + 1);
    mq.addEventListener("change", bump);
    const mo = new MutationObserver(bump);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => { mq.removeEventListener("change", bump); mo.disconnect(); };
  }, []);
  return v;
}

export function usePalette() {
  const v = useThemeVersion();
  // the version only exists to re-run this on theme change
  void v;
  return {
    s1: tok("--s1"), s2: tok("--s2"), s3: tok("--s3"), s4: tok("--s4"), s5: tok("--s5"), s6: tok("--s6"),
    good: tok("--good"), bad: tok("--bad"),
    ink: tok("--ink"), ink2: tok("--ink-2"), muted: tok("--muted"), rule: tok("--rule-2"), surface: tok("--chart-surface"),
  };
}

// ------------------------------------------------------------- wrapper -----
export function Chart({ option, height = 260, onClick, label }) {
  const el = useRef(null);
  const inst = useRef(null);
  const clickRef = useRef(onClick);
  clickRef.current = onClick;
  useEffect(() => {
    inst.current = echarts.init(el.current, null, { renderer: "canvas" });
    inst.current.on("click", (params) => clickRef.current && clickRef.current(params));
    const ro = new ResizeObserver(() => inst.current && inst.current.resize());
    ro.observe(el.current);
    return () => { ro.disconnect(); if (inst.current) { inst.current.dispose(); inst.current = null; } };
  }, []);
  useEffect(() => { if (inst.current && option) inst.current.setOption(option, { notMerge: true }); }, [option]);
  return (
    <div ref={el} className={"chartbox" + (onClick ? " clickable" : "")} style={{ height }}
         role="img" aria-label={label || "chart"} />
  );
}

// --------------------------------------------------------- option parts ----
const FONT = '"IBM Plex Sans", system-ui, sans-serif';
const MONO = '"IBM Plex Mono", monospace';

export const fmt = {
  comma: (v) => Math.round(v).toLocaleString(),
  pct: (v) => `${v}%`,
  pct1: (v) => `${Number(v).toFixed(1)}%`,
  frac: (v) => `${Math.round(v * 100)}%`,
  fixed1: (v) => Number(v).toFixed(1),
};

function base(p, { legend = false, legendNames = [] } = {}) {
  return {
    backgroundColor: "transparent",
    textStyle: { fontFamily: FONT, color: p.ink2 },
    animationDuration: 300,
    grid: { left: 8, right: 16, top: legend ? 38 : 14, bottom: 6, containLabel: true },
    legend: legend
      ? { show: true, top: 0, left: 0, icon: "circle", itemWidth: 9, itemHeight: 9, itemGap: 16,
          textStyle: { color: p.ink2, fontSize: 12 }, data: legendNames }
      : { show: false },
    tooltip: {
      backgroundColor: p.surface, borderColor: p.rule, borderWidth: 1, padding: [8, 10],
      textStyle: { color: p.ink, fontFamily: FONT, fontSize: 12 },
      extraCssText: "box-shadow: 0 4px 16px rgba(0,0,0,.12); border-radius: 6px;",
    },
  };
}

function categoryAxis(p, data, extra = {}) {
  return {
    type: "category", data, boundaryGap: extra.bars ?? false,
    axisLine: { lineStyle: { color: p.rule } }, axisTick: { show: false },
    axisLabel: { color: p.muted, fontSize: 11, hideOverlap: true, ...extra.axisLabel },
    splitLine: { show: false },
  };
}
function valueAxis(p, { log = false, formatter, min, max } = {}) {
  return {
    type: log ? "log" : "value", logBase: 10, min, max,
    axisLine: { show: false }, axisTick: { show: false },
    axisLabel: { color: p.muted, fontSize: 11, formatter },
    splitLine: { lineStyle: { color: p.rule } },
  };
}

function lineSeries(s) {
  return {
    type: "line", name: s.name, data: s.data, showSymbol: false, symbol: "circle", symbolSize: 8,
    lineStyle: { width: 2, color: s.color, type: s.dash ? "dashed" : "solid" },
    itemStyle: { color: s.color, borderColor: "#fff", borderWidth: 2 },
    emphasis: { focus: "series", lineStyle: { width: 2.5 } }, smooth: 0.15,
  };
}
function barSeries(s, { horizontal = false, stack = null, maxWidth = 30 } = {}) {
  return {
    type: "bar", name: s.name, data: s.data, stack: stack || undefined, barMaxWidth: maxWidth,
    barGap: stack ? "0%" : "15%", barCategoryGap: "35%",
    itemStyle: { color: s.color, borderRadius: stack ? 0 : (horizontal ? [0, 3, 3, 0] : [3, 3, 0, 0]),
                 borderColor: stack ? "transparent" : undefined, borderWidth: stack ? 1 : 0 },
    emphasis: { focus: "series" },
  };
}

// ------------------------------------------------------------- builders ----
/** Time-series line chart. series: [{name, data, color}] aligned with labels. */
export function lineOption(p, { labels, series, log = false, fmtY }) {
  const multi = series.length > 1;
  const clean = series.map((s) => ({ ...s, data: log ? s.data.map((v) => (v > 0 ? v : null)) : s.data }));
  return {
    ...base(p, { legend: multi, legendNames: series.map((s) => s.name) }),
    tooltip: { ...base(p).tooltip, trigger: "axis", axisPointer: { type: "line", lineStyle: { color: p.rule } },
               valueFormatter: (v) => (v == null ? "—" : fmtY ? fmtY(v) : v) },
    xAxis: categoryAxis(p, labels),
    yAxis: valueAxis(p, { log, formatter: fmtY }),
    series: clean.map(lineSeries),
  };
}

/** Vertical bars (grouped or stacked). */
export function barOption(p, { labels, series, stacked = false, fmtY }) {
  const multi = series.length > 1;
  return {
    ...base(p, { legend: multi, legendNames: series.map((s) => s.name) }),
    tooltip: { ...base(p).tooltip, trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: "rgba(128,128,128,.08)" } },
               valueFormatter: (v) => (v == null ? "—" : fmtY ? fmtY(v) : v) },
    xAxis: categoryAxis(p, labels, { bars: true }),
    yAxis: valueAxis(p, { formatter: fmtY }),
    series: series.map((s) => barSeries(s, { stack: stacked ? "s" : null })),
  };
}

/** Horizontal bars, largest at the top. labels/series data are given top-to-bottom. */
export function hbarOption(p, { labels, series, stacked = false, fmtX, labelWidth = 150, max }) {
  const multi = series.length > 1;
  const rev = (a) => [...a].reverse();
  return {
    ...base(p, { legend: multi, legendNames: series.map((s) => s.name) }),
    grid: { left: 8, right: 44, top: multi ? 38 : 8, bottom: 6, containLabel: true },
    tooltip: { ...base(p).tooltip, trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: "rgba(128,128,128,.08)" } },
               valueFormatter: (v) => (v == null ? "—" : fmtX ? fmtX(v) : v) },
    xAxis: valueAxis(p, { formatter: fmtX, max }),
    yAxis: categoryAxis(p, rev(labels), { bars: true, axisLabel: { color: p.ink2, fontSize: 11.5, width: labelWidth, overflow: "truncate" } }),
    series: series.map((s) => barSeries({ ...s, data: rev(s.data) }, { horizontal: true, stack: stacked ? "s" : null, maxWidth: 18 })),
  };
}

/** Log-log "share with at least x" curve. data: [{x, share}] */
export function loglogOption(p, { name, color, data, xLabel }) {
  return {
    ...base(p),
    grid: { left: 8, right: 20, top: 16, bottom: 28, containLabel: true },
    tooltip: { ...base(p).tooltip, trigger: "axis", axisPointer: { type: "line", lineStyle: { color: p.rule } },
      formatter: (params) => { const d = params[0]?.data; if (!d) return "";
        const pctStr = d[1] < 0.001 ? (d[1] * 100).toFixed(4) : (d[1] * 100).toFixed(2);
        return `<b>${pctStr}%</b> have ≥ ${Math.round(d[0]).toLocaleString()}`; } },
    xAxis: { type: "log", logBase: 10, name: xLabel, nameLocation: "middle", nameGap: 26,
             nameTextStyle: { color: p.muted, fontSize: 11.5 },
             axisLine: { lineStyle: { color: p.rule } }, axisTick: { show: false },
             axisLabel: { color: p.muted, fontSize: 11, formatter: (v) => Number(v).toLocaleString() },
             splitLine: { lineStyle: { color: p.rule } } },
    yAxis: { type: "log", logBase: 10, axisLine: { show: false }, axisTick: { show: false },
             axisLabel: { color: p.muted, fontSize: 11, formatter: (v) => { const s = v * 100; return (s >= 1 ? s.toFixed(0) : s.toPrecision(1)) + "%"; } },
             splitLine: { lineStyle: { color: p.rule } } },
    series: [{ type: "line", name, data: data.map((r) => [r.x, r.share]), showSymbol: false,
               lineStyle: { width: 2, color }, itemStyle: { color }, smooth: false }],
  };
}

/** Growth-rate bars with confidence intervals in the tooltip. rows: [{kind, period, rate, lo, hi, doubling}] */
export function growthOption(p, rows) {
  const kinds = ["all", "journal", "conference", "preprint"];
  const periods = [...new Set(rows.map((r) => r.period))].slice(0, 2);
  const colors = [p.s4, p.s6];
  const find = (k, per) => rows.find((r) => r.kind === k && r.period === per);
  return {
    ...base(p, { legend: true, legendNames: periods }),
    tooltip: { ...base(p).tooltip, trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: "rgba(128,128,128,.08)" } },
      formatter: (params) => params.map((q) => { const r = find(kinds[q.dataIndex], q.seriesName); if (!r) return "";
        const doubling = r.doubling ? `, doubles in ${r.doubling} y` : "";
        return `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${q.color};margin-right:6px"></span>` +
               `${q.seriesName}: <b>${r.rate}%/yr</b> <span style="color:${p.muted}">(${r.lo}–${r.hi}%${doubling})</span>`; }).join("<br>") },
    xAxis: categoryAxis(p, kinds.map((k) => k[0].toUpperCase() + k.slice(1)), { bars: true }),
    yAxis: valueAxis(p, { formatter: (v) => v + "%" }),
    series: periods.map((per, i) => barSeries({ name: per, color: colors[i], data: kinds.map((k) => find(k, per)?.rate ?? null) }, { maxWidth: 26 })),
  };
}
