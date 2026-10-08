import { useEffect, useRef, useState } from "react";
import * as echarts from "echarts";
import { contours as d3contours } from "d3-contour";

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
    seq: [tok("--seq-1"), tok("--seq-2"), tok("--seq-3"), tok("--seq-4"), tok("--seq-5")],
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
               valueFormatter: (v) => (v == null ? "–" : fmtY ? fmtY(v) : v) },
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
               valueFormatter: (v) => (v == null ? "–" : fmtY ? fmtY(v) : v) },
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
               valueFormatter: (v) => (v == null ? "–" : fmtX ? fmtX(v) : v) },
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

/**
 * Force-directed network. nodes: [{id, name, value, category, hollow?, label?}], links: [{source, target, value, dashed?}],
 * categories: [{name, color}]. Node size follows sqrt(value) so area is proportional; every mark wears a 2px surface ring.
 */
export function graphOption(p, { nodes, links, categories, sizeRange = [10, 34], tooltip }) {
  const vals = nodes.map((n) => n.value || 0);
  const lo = Math.sqrt(Math.max(1, Math.min(...vals))), hi = Math.sqrt(Math.max(1, ...vals));
  const size = (v) => (hi === lo ? sizeRange[0] : sizeRange[0] + ((Math.sqrt(Math.max(1, v)) - lo) / (hi - lo)) * (sizeRange[1] - sizeRange[0]));
  const maxLink = Math.max(1, ...links.map((l) => l.value || 1));
  return {
    ...base(p, { legend: true, legendNames: categories.map((c) => c.name) }),
    legend: { show: true, top: 0, left: 0, icon: "circle", itemWidth: 9, itemHeight: 9, itemGap: 14,
              textStyle: { color: p.ink2, fontSize: 12 }, data: categories.map((c) => c.name) },
    tooltip: { ...base(p).tooltip, trigger: "item", formatter: tooltip },
    series: [{
      type: "graph", layout: "force", roam: true, draggable: true, zoom: 1,
      force: { repulsion: 320, gravity: 0.12, edgeLength: [50, 150], friction: 0.15 },
      categories: categories.map((c) => ({ name: c.name, itemStyle: { color: c.color } })),
      data: nodes.map((n) => ({
        id: n.id, name: n.name, value: n.value, category: n.category, symbolSize: size(n.value),
        itemStyle: n.hollow
          ? { color: p.surface, borderColor: categories[n.category].color, borderWidth: 2, borderType: "dashed" }
          : { borderColor: p.surface, borderWidth: 2 },
        // a surface-coloured plate keeps a label legible where it crosses another node
        label: { show: !!n.label, position: "right", color: p.ink2, fontSize: 11, fontFamily: FONT,
                 backgroundColor: p.surface, padding: [1, 4], borderRadius: 3,
                 formatter: (d) => d.name.length > 22 ? d.name.slice(0, 21) + "…" : d.name },
      })),
      links: links.map((l) => ({
        source: l.source, target: l.target, value: l.value,
        lineStyle: { width: 1 + 2.5 * Math.sqrt((l.value || 1) / maxLink), color: p.rule, curveness: 0,
                     type: l.dashed ? "dashed" : "solid", opacity: l.dashed ? 0.9 : 0.8 },
      })),
      lineStyle: { color: p.rule },
      emphasis: { focus: "adjacency", label: { show: true }, lineStyle: { width: 3 } },
      labelLayout: { hideOverlap: true },
    }],
  };
}

/**
 * Radar: axes share one scale (0-100) so shapes are comparable; a reference series is drawn as a dashed gray
 * polygon with no fill. series: [{name, values, color, reference?}], axes: [{name}].
 */
export function radarOption(p, { axes, series, tooltip }) {
  return {
    ...base(p, { legend: true, legendNames: series.map((s) => s.name) }),
    tooltip: { ...base(p).tooltip, trigger: "item", formatter: tooltip },
    radar: {
      indicator: axes.map((a) => ({ name: a.name, max: 100 })), radius: "64%", center: ["50%", "56%"],
      shape: "polygon", splitNumber: 4,
      axisName: { color: p.ink2, fontSize: 11, fontFamily: FONT },
      axisLine: { lineStyle: { color: p.rule } },
      splitLine: { lineStyle: { color: p.rule } },
      splitArea: { show: false },
    },
    series: [{
      type: "radar", symbol: "circle", symbolSize: 8,
      data: series.map((s) => ({
        name: s.name, value: s.values,
        lineStyle: { width: 2, color: s.color, type: s.reference ? "dashed" : "solid" },
        itemStyle: { color: s.color, borderColor: p.surface, borderWidth: 2 },
        areaStyle: s.reference ? { opacity: 0 } : { color: s.color, opacity: 0.12 },
      })),
    }],
  };
}

/**
 * Two-level treemap: top level coloured categorically in fixed order (a 7th+ group folds to gray), leaves within
 * a group vary by alpha so bigger reads darker. children: [{name, value, children: [{name, value, sid}]}].
 */
export function treemapOption(p, { children, tooltip }) {
  const hues = [p.s1, p.s2, p.s3, p.s4, p.s5, p.s6];
  return {
    ...base(p),
    tooltip: { ...base(p).tooltip, trigger: "item", formatter: tooltip },
    series: [{
      type: "treemap", roam: false, nodeClick: false, width: "100%", height: "100%", top: 28, left: 0, right: 0, bottom: 0,
      breadcrumb: { show: false },
      upperLabel: { show: true, height: 22, color: p.ink, fontSize: 12, fontWeight: 600, fontFamily: FONT,
                    formatter: (d) => d.name },
      label: { show: true, color: p.ink, fontSize: 11, fontFamily: FONT, overflow: "truncate",
               formatter: (d) => d.name },
      levels: [
        { itemStyle: { gapWidth: 3, borderWidth: 0, borderColor: p.surface }, upperLabel: { show: false } },
        { itemStyle: { gapWidth: 3, borderWidth: 3, borderColor: p.surface }, colorAlpha: [0.8, 1], upperLabel: { show: true } },
        { itemStyle: { gapWidth: 2, borderWidth: 2, borderColor: p.surface }, colorAlpha: [0.45, 0.95] },
      ],
      data: children.map((c, i) => ({
        ...c, itemStyle: { color: i < hues.length ? hues[i] : p.muted },
        children: c.children.map((leaf) => ({ ...leaf, itemStyle: { color: i < hues.length ? hues[i] : p.muted } })),
      })),
    }],
  };
}

/**
 * Scatter with a log x axis. series: [{name, color, data: [{x, y, ...meta}]}]; the `labelTop` largest-x points
 * are direct-labelled. Points wear a surface ring so overlaps stay countable.
 */
export function scatterOption(p, { series, xName, yName, fmtX, fmtY, labelTop = 8, tooltip }) {
  const all = series.flatMap((s) => s.data);
  const labelled = new Set([...all].sort((a, b) => b.x - a.x).slice(0, labelTop).map((d) => d.name));
  return {
    ...base(p, { legend: series.length > 1, legendNames: series.map((s) => s.name) }),
    grid: { left: 44, right: 24, top: series.length > 1 ? 38 : 16, bottom: 30, containLabel: true },
    tooltip: { ...base(p).tooltip, trigger: "item", formatter: tooltip },
    xAxis: { type: "log", logBase: 10, name: xName, nameLocation: "middle", nameGap: 26,
             nameTextStyle: { color: p.muted, fontSize: 11.5 },
             axisLine: { lineStyle: { color: p.rule } }, axisTick: { show: false },
             axisLabel: { color: p.muted, fontSize: 11, formatter: fmtX }, splitLine: { lineStyle: { color: p.rule } } },
    yAxis: { type: "value", name: yName, nameLocation: "middle", nameGap: 40, nameRotate: 90, nameTextStyle: { color: p.muted, fontSize: 11.5 },
             axisLine: { show: false }, axisTick: { show: false },
             axisLabel: { color: p.muted, fontSize: 11, formatter: fmtY }, splitLine: { lineStyle: { color: p.rule } } },
    series: series.map((s) => ({
      type: "scatter", name: s.name, symbolSize: 9,
      itemStyle: { color: s.color, borderColor: p.surface, borderWidth: 1.5, opacity: 0.9 },
      emphasis: { focus: "series", itemStyle: { opacity: 1 } },
      label: { show: true, position: "right", color: p.ink2, fontSize: 10.5, fontFamily: FONT,
               formatter: (d) => (labelled.has(d.data.name) ? d.data.name : "") },
      labelLayout: { hideOverlap: true },
      data: s.data.map((d) => ({ ...d, value: [d.x, d.y] })),
    })),
  };
}

/**
 * The research page's figure: per model, the share of the oracle's gain that RAG recovers against how
 * often retrieval puts the right paper in the context, with the line where the two are equal.
 * series: [{name, color, data: [{x (0-100), y (0-1), where}]}]
 */
export function recoveryOption(p, { series, xName, yName }) {
  const names = [...series.map((s) => s.name), "equal shares"];
  return {
    ...base(p, { legend: true, legendNames: names }),
    grid: { left: 44, right: 22, top: 40, bottom: 34, containLabel: true },
    tooltip: { ...base(p).tooltip, trigger: "item",
      formatter: (d) => `<b>${d.seriesName}</b> · ${d.data.where}<br/>right paper in the context: ${d.data.value[0]}%`
        + `<br/>share of the right abstract's gain recovered: ${Math.round(d.data.value[1] * 100)}%` },
    xAxis: { type: "value", min: 0, max: 100, interval: 20, name: xName, nameLocation: "middle", nameGap: 26,
             nameTextStyle: { color: p.muted, fontSize: 11.5 },
             axisLine: { lineStyle: { color: p.rule } }, axisTick: { show: false },
             axisLabel: { color: p.muted, fontSize: 11, formatter: (v) => `${v}%` }, splitLine: { lineStyle: { color: p.rule } } },
    yAxis: { type: "value", min: 0, max: 1, interval: 0.2, name: yName, nameLocation: "middle", nameGap: 40, nameRotate: 90,
             nameTextStyle: { color: p.muted, fontSize: 11.5 }, axisLine: { show: false }, axisTick: { show: false },
             axisLabel: { color: p.muted, fontSize: 11, formatter: fmt.frac }, splitLine: { lineStyle: { color: p.rule } } },
    series: [
      ...series.map((s) => ({
        type: "line", name: s.name, symbol: "circle", symbolSize: 9,
        data: s.data.map((d) => ({ value: [d.x, d.y], where: d.where })),
        lineStyle: { width: 2, color: s.color }, itemStyle: { color: s.color, borderColor: p.surface, borderWidth: 2 },
        emphasis: { focus: "series", lineStyle: { width: 2.5 } },
      })),
      { type: "line", name: "equal shares", data: [[0, 0], [100, 1]], symbol: "none", silent: true,
        tooltip: { show: false }, itemStyle: { color: p.muted },
        lineStyle: { color: p.muted, width: 1, type: "dashed" } },
    ],
  };
}

/**
 * 2-D density on log-log axes: a grid of log10 bins drawn as cells on a sequential ramp (colour is log of the
 * count, since counts span many orders of magnitude), with iso-count contours (marching squares) on top.
 * grid[y][x] = count; bin i covers 10^(i/b) .. 10^((i+1)/b).
 */
export function densityOption(p, { grid, nx, ny, binsPerDecade, xName, yName, levels, total }) {
  const b = binsPerDecade;
  const lo = (i) => Math.pow(10, i / b), mid = (i) => Math.pow(10, (i + 0.5) / b);
  const maxLog = Math.log10(Math.max(1, ...grid.flat()) + 1);
  const ramp = p.seq;
  const colorAt = (v) => ramp[Math.min(ramp.length - 1, Math.floor((Math.log10(v + 1) / maxLog) * ramp.length))];
  const cells = [];
  grid.forEach((row, y) => row.forEach((v, x) => { if (v > 0) cells.push([x, y, v]); }));
  // contours on the log-count field, in grid coordinates, then mapped to axis values through bin centres
  const field = new Float64Array(nx * ny);
  grid.forEach((row, y) => row.forEach((v, x) => { field[y * nx + x] = Math.log10(v + 1); }));
  const rings = d3contours().size([nx, ny]).thresholds(levels.map((l) => Math.log10(l + 1)))(field)
    .flatMap((c, li) => c.coordinates.flatMap((poly) => poly.map((ring) => ({
      level: levels[li], coords: ring.map(([gx, gy]) => [mid(gx - 0.5), mid(gy - 0.5)]) }))));
  const axis = (name) => ({
    type: "log", logBase: 10, name, nameLocation: "middle", nameGap: 28, nameTextStyle: { color: p.muted, fontSize: 11.5 },
    axisLine: { lineStyle: { color: p.rule } }, axisTick: { show: false },
    axisLabel: { color: p.muted, fontSize: 11, formatter: (v) => Number(v).toLocaleString() },
    splitLine: { lineStyle: { color: p.rule } },
  });
  return {
    ...base(p),
    // containLabel covers tick labels, not axis names: the y name needs its own room on the left
    grid: { left: 44, right: 20, top: 16, bottom: 30, containLabel: true },
    tooltip: { ...base(p).tooltip, trigger: "item", formatter: (q) => {
      if (q.seriesType !== "custom") return "";
      const [x, y, v] = q.value;
      const r = (i) => `${Math.ceil(lo(i)).toLocaleString()}–${Math.floor(lo(i + 1) - 1e-9).toLocaleString()}`;
      return `${xName}: <b>${r(x)}</b><br>${yName}: <b>${r(y)}</b><br><b>${v.toLocaleString()}</b> authors (${(100 * v / total).toFixed(2)}%)`; } },
    xAxis: { ...axis(xName), min: 1, max: lo(nx) },
    yAxis: { ...axis(yName), nameGap: 40, nameRotate: 90, min: 1, max: lo(ny) },
    series: [
      { type: "custom", name: "authors", z: 1,
        renderItem: (params, api) => {
          const x = api.value(0), y = api.value(1), v = api.value(2);
          const a = api.coord([lo(x), lo(y)]), c = api.coord([lo(x + 1), lo(y + 1)]);
          return { type: "rect", shape: { x: a[0], y: c[1], width: c[0] - a[0] - 1, height: a[1] - c[1] - 1 },
                   style: { fill: colorAt(v) } };
        },
        encode: { x: 0, y: 1 }, data: cells },
      { type: "lines", coordinateSystem: "cartesian2d", polyline: true, silent: true, z: 2,
        lineStyle: { color: p.ink2, width: 1.5, opacity: 0.9 },
        data: rings.map((r) => ({ coords: r.coords, level: r.level })) },
    ],
  };
}

/** "#rrggbb" -> "rgba(r,g,b,a)": a translucent fill without dimming the strokes drawn on top of it. */
function alpha(hex, a) {
  const n = parseInt(hex.replace("#", ""), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/** Boxplots per group: box = quartiles, whiskers = 5th/95th percentile; the mean as a small marker. */
export function boxplotOption(p, { labels, boxes, means, fmtY }) {
  return {
    ...base(p, { legend: true, legendNames: ["Quartiles, 5th–95th percentile", "Mean"] }),
    tooltip: { ...base(p).tooltip, trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: "rgba(128,128,128,.08)" } },
      formatter: (params) => {
        const box = params.find((q) => q.seriesType === "boxplot"), mean = params.find((q) => q.seriesType === "scatter");
        if (!box) return "";
        const [, p5, q1, med, q3, p95] = box.data;
        return `<b>${box.name}</b><br>median <b>${med}</b> · quartiles ${q1}–${q3}<br>5th–95th percentile ${p5}–${p95}` +
               (mean ? `<br>mean ${Number(mean.data[1]).toFixed(2)}` : ""); } },
    xAxis: categoryAxis(p, labels, { bars: true }),
    yAxis: valueAxis(p, { formatter: fmtY }),
    series: [
      { type: "boxplot", name: "Quartiles, 5th–95th percentile", data: boxes, boxWidth: [12, 34],
        itemStyle: { color: alpha(p.s1, 0.18), borderColor: p.s1, borderWidth: 2 },
        emphasis: { itemStyle: { color: alpha(p.s1, 0.3) } } },
      { type: "scatter", name: "Mean", data: means.map((m, i) => [i, m]), symbolSize: 8,
        itemStyle: { color: p.ink, borderColor: p.surface, borderWidth: 2 } },
    ],
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
