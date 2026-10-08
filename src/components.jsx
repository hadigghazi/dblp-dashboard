import { useMemo, useState } from "react";
import { tok, useThemeVersion } from "./charts.jsx";
import { fmtDate } from "./api.js";

export function Kpi({ n, l }) {
  return (
    <div className="kpi">
      <div className="n num">{n}</div>
      <div className="l">{l}</div>
    </div>
  );
}
export function KpiStrip({ items, loading }) {
  if (!items) {
    return <div className="kpis">{Array.from({ length: loading ? 4 : 0 }, (_, i) => <div key={i} className="kpi"><div className="skel" style={{ height: 22, width: "60%" }} /><div className="skel" style={{ height: 12, marginTop: 8 }} /></div>)}</div>;
  }
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

/** Where a card's numbers come from: computed now from the dump, or read from an analysis job's output. */
export function SourceTag({ job }) {
  if (!job) return <span className="srctag live" title="Computed on request from dblp.parquet">Live</span>;
  return (
    <span className="srctag job" title={`Read from ${job.file}, the output of the analysis job`}>
      Job output · {fmtDate(job.modified)}
    </span>
  );
}

/**
 * A card. Pass `state` (from useApi) to get loading, warming-up and error handling for free;
 * children render once data exists (they receive it via the render prop when `children` is a function).
 */
export function Card({ id, title, sub, span2, state, job, height = 260, controls, children }) {
  const hasState = state !== undefined;
  const data = hasState ? state.data : undefined;
  let body;
  if (hasState && state.error) {
    body = <div className="cardmsg error" role="alert"><b>Couldn’t load this.</b> {state.error}</div>;
  } else if (hasState && !data) {
    body = state.warming
      ? <div className="cardmsg" style={{ minHeight: height }}><Spinner /> {state.warming.message || "Preparing live data…"}</div>
      : <div className="skel" style={{ height, marginTop: 10 }} />;
  } else {
    body = typeof children === "function" ? children(data) : children;
  }
  return (
    <section id={id} className={"card" + (span2 ? " span2" : "")} aria-busy={hasState && state.loading}>
      <div className="cardhead">
        <h3>{title}</h3>
        <div className="cardtags">
          {hasState && state.loading && data ? <span className="updating">Updating…</span> : null}
          {hasState || job ? <SourceTag job={job} /> : null}
        </div>
      </div>
      {sub ? <div className="cardsub">{sub}</div> : null}
      {controls ? <div className="filters">{controls}</div> : null}
      <div className="body">{body}</div>
    </section>
  );
}

export function Spinner() {
  return <span className="spinner" aria-hidden="true" />;
}

export function Filters({ page, children }) {
  return <div className={"filters" + (page ? " page" : "")}>{children}</div>;
}
export function FilterLabel({ children, htmlFor }) {
  return htmlFor ? <label className="filterlabel" htmlFor={htmlFor}>{children}</label> : <span className="filterlabel">{children}</span>;
}

export function Chip({ label, on, color, onClick }) {
  return (
    <button type="button" className={"chip" + (on ? " on" : "")} onClick={onClick} aria-pressed={on}>
      {color ? <span className="dot" style={{ background: on ? color : "var(--muted)" }} /> : null}
      {label}
    </button>
  );
}
export function ChipGroup({ children }) { return <div className="chipgroup">{children}</div>; }

export function Seg({ options, value, onChange, label }) {
  return (
    <div className="segtoggle" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={o.v} type="button" role="radio" aria-checked={value === o.v}
                className={value === o.v ? "on" : ""} onClick={() => onChange(o.v)}>{o.l}</button>
      ))}
    </div>
  );
}

export function RangeSlider({ id, min, max, value, onChange, format, label }) {
  return (
    <div className="rangewrap">
      <span className="num">{format ? format(value) : value}</span>
      <input id={id} type="range" min={min} max={max} value={value} aria-label={label}
             onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}

export function NumberInput({ id, value, onChange, min, max, step = 1, width = 76, label }) {
  return (
    <input id={id} className="numinput num" type="number" value={value} min={min} max={max} step={step}
           aria-label={label} style={{ width }}
           onChange={(e) => e.target.value !== "" && onChange(+e.target.value)} />
  );
}

export function SearchBox({ id, value, onChange, placeholder, autoFocus }) {
  return (
    <div className="searchbox">
      <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true"><circle cx="8.5" cy="8.5" r="5.5" fill="none" stroke="currentColor" strokeWidth="2" /><path d="M13 13l4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
      <input id={id} type="search" value={value} placeholder={placeholder} autoFocus={autoFocus}
             aria-label={placeholder} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

const KIND_LABEL = { regular: "Author page", numbered: "Numbered namesake", disambiguation: "Disambiguation bin", unresolved: "No author page" };
export function KindBadge({ kind }) {
  return <span className={`kbadge k-${kind}`}>{KIND_LABEL[kind] || kind}</span>;
}

export function PubKind({ kind }) {
  return <span className={`pkind p-${kind}`}>{kind}</span>;
}

export function SortableTable({ columns, rows, defaultSort, onRowClick }) {
  const [sort, setSort] = useState(defaultSort || { key: columns[0].key, dir: -1 });
  const sorted = useMemo(() => {
    const arr = [...rows];
    arr.sort((a, b) => {
      const av = a[sort.key], bv = b[sort.key];
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * sort.dir;
      return String(av ?? "").localeCompare(String(bv ?? "")) * sort.dir;
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
                  aria-sort={sort.key === c.key ? (sort.dir === -1 ? "descending" : "ascending") : "none"}>
                <button type="button" className="thbtn" onClick={() => setSort((s) => ({ key: c.key, dir: s.key === c.key ? -s.dir : -1 }))}>
                  {c.label}{sort.key === c.key ? <span className="arrow">{sort.dir === -1 ? "▼" : "▲"}</span> : null}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={r.key || i} className={onRowClick ? "clickable" : ""} onClick={onRowClick ? () => onRowClick(r) : undefined}>
              {columns.map((c) => <td key={c.key} className={c.num ? "num" : ""}>{c.render ? c.render(r) : r[c.key]}</td>)}
            </tr>
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
          <tr><th />{colLabels.map((c, i) => <th key={i} style={{ textAlign: "center" }}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rowLabels.map((r, ri) => (
            <tr key={ri}>
              <td style={{ color: "var(--ink)", fontWeight: 500, textAlign: "left" }}>{r}</td>
              {matrix[ri].map((val, ci) => (
                <td key={ci} className="num" title={`${r} · ${colLabels[ci]}: ${val == null ? "n/a" : (val * 100).toFixed(1) + "%"}`}
                    style={{ background: `rgba(${rgb}, ${Math.max(0.05, val || 0)})`, color: val > 0.55 ? "#fff" : "var(--ink-2)" }}>
                  {val == null ? "–" : Math.round(val * 100) + "%"}
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

export function EmptyNote({ children }) { return <div className="cardmsg">{children}</div>; }

/** Internal link to another page of the app (hash routing). */
export function Link({ to, params, children, className }) {
  const qs = new URLSearchParams(params || {}).toString();
  return <a className={className} href={`#${to}${qs ? `?${qs}` : ""}`}>{children}</a>;
}
