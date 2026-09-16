import { useMemo, useState } from "react";
import { tok, useThemeVersion } from "./charts.jsx";

export function Kpi({ n, l }) {
  return (
    <div className="kpi">
      <div className="n num">{n}</div>
      <div className="l">{l}</div>
    </div>
  );
}
export function KpiStrip({ items }) {
  return <div className="kpis">{items.map((k, i) => <Kpi key={i} {...k} />)}</div>;
}

export function PageHead({ eyebrow, title, children }) {
  return (
    <div className="pagehead">
      <div className="eyebrow">{eyebrow}</div>
      <h2>{title}</h2>
      {children ? <p>{children}</p> : null}
    </div>
  );
}

export function Card({ title, sub, span2, children }) {
  return (
    <div className={"card" + (span2 ? " span2" : "")}>
      <h3>{title}</h3>
      {sub ? <div className="cardsub">{sub}</div> : null}
      <div className="body">{children}</div>
    </div>
  );
}

export function Filters({ page, children }) {
  return <div className={"filters" + (page ? " page" : "")}>{children}</div>;
}
export function FilterLabel({ children }) { return <span className="filterlabel">{children}</span>; }

export function Chip({ label, on, color, onClick }) {
  return (
    <button type="button" className={"chip" + (on ? " on" : "")} onClick={onClick} aria-pressed={on}>
      {color ? <span className="dot" style={{ background: on ? color : "var(--muted)" }} /> : null}
      {label}
    </button>
  );
}
export function ChipGroup({ children }) { return <div className="chipgroup">{children}</div>; }

export function Seg({ options, value, onChange }) {
  return (
    <div className="segtoggle" role="tablist">
      {options.map((o) => (
        <button key={o.v} type="button" role="tab" aria-selected={value === o.v}
                className={value === o.v ? "on" : ""} onClick={() => onChange(o.v)}>{o.l}</button>
      ))}
    </div>
  );
}

export function RangeSlider({ id, min, max, value, onChange, format }) {
  return (
    <div className="rangewrap">
      <span className="num">{format ? format(value) : value}</span>
      <input id={id} type="range" min={min} max={max} value={value} onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}

export function SortableTable({ columns, rows, defaultSort }) {
  const [sort, setSort] = useState(defaultSort || { key: columns[0].key, dir: -1 });
  const sorted = useMemo(() => {
    const arr = [...rows];
    arr.sort((a, b) => {
      const av = a[sort.key], bv = b[sort.key];
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * sort.dir;
      return String(av).localeCompare(String(bv)) * sort.dir;
    });
    return arr;
  }, [rows, sort]);
  return (
    <div className="tablewrap">
      <table className="data">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} style={{ textAlign: c.num ? "right" : "left", paddingRight: c.num ? 18 : 0 }}
                  onClick={() => setSort((s) => ({ key: c.key, dir: s.key === c.key ? -s.dir : -1 }))}>
                {c.label}{sort.key === c.key ? <span className="arrow">{sort.dir === -1 ? "▼" : "▲"}</span> : null}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={i}>{columns.map((c) => <td key={c.key} className={c.num ? "num" : ""}>{c.render ? c.render(r) : r[c.key]}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || "");
  return m ? `${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)}` : "42,120,214";
}

/** A coverage matrix rendered as a table so it stays theme-aware and readable without a canvas. */
export function Heatmap({ rowLabels, colLabels, matrix }) {
  const v = useThemeVersion();
  const rgb = useMemo(() => hexToRgb(tok("--s1")), [v]);
  return (
    <div className="tablewrap">
      <table className="data heat" style={{ fontSize: 11.5 }}>
        <thead>
          <tr><th style={{ cursor: "default" }} /> {colLabels.map((c, i) => <th key={i} style={{ textAlign: "center", cursor: "default" }}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rowLabels.map((r, ri) => (
            <tr key={ri}>
              <td style={{ color: "var(--ink)", fontWeight: 500, textAlign: "left" }}>{r}</td>
              {matrix[ri].map((val, ci) => (
                <td key={ci} className="num" style={{ background: `rgba(${rgb}, ${Math.max(0.05, val)})`, color: val > 0.55 ? "#fff" : "var(--ink-2)" }}>
                  {val == null ? "—" : Math.round(val * 100) + "%"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Callout({ children }) { return <div className="callout">{children}</div>; }
