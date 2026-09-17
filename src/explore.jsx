import { useEffect, useMemo, useState } from "react";
import { useApi, useDebounced } from "./api.js";
import { Chart, usePalette, barOption, lineOption, fmt } from "./charts.jsx";
import {
  KpiStrip, PageHead, Card, Filters, FilterLabel, Seg, NumberInput, SearchBox, SortableTable,
  KindBadge, PubKind, Callout, EmptyNote,
} from "./components.jsx";
import { BinSplit } from "./ml.jsx";

const KINDS = ["journal", "conference", "preprint", "phdthesis", "book", "incollection", "data", "mastersthesis"];
const KIND_COLORS = (p) => ({ journal: p.s2, conference: p.s1, preprint: p.s3, phdthesis: p.s6, book: p.s4, incollection: p.s5, data: p.muted, mastersthesis: p.muted });

/** Search text kept in the URL (replace, not push) so back/forward and shared links work. */
function useQueryParam(params, go, page, name) {
  const [value, setValue] = useState(params[name] || "");
  useEffect(() => { setValue(params[name] || ""); }, [params[name]]);
  const debounced = useDebounced(value, 400);
  useEffect(() => {
    // a new search replaces whatever detail was open
    if ((params[name] || "") !== debounced) go(page, debounced ? { [name]: debounced } : {}, { replace: true });
  }, [debounced]);
  return [value, setValue, debounced];
}

function yearsPivot(rows, key = "kind") {
  const years = [...new Set(rows.map((r) => r.year))].sort((a, b) => a - b);
  if (!years.length) return { years: [], kinds: [], get: () => [] };
  const all = [];
  for (let y = years[0]; y <= years[years.length - 1]; y++) all.push(y);
  const kinds = [...new Set(rows.map((r) => r[key]))];
  const m = new Map(rows.map((r) => [`${r.year}|${r[key]}`, r.papers]));
  return { years: all, kinds, get: (k) => all.map((y) => m.get(`${y}|${k}`) || 0) };
}

// ============================================================== Authors ====
export function PageAuthors({ params, go }) {
  const [q, setQ, qd] = useQueryParam(params, go, "authors", "q");
  const key = params.key;
  const search = useApi(qd.trim().length >= 2 ? "authors/search" : null, { q: qd.trim() });
  return (
    <>
      <PageHead eyebrow="Explore · Authors" title="Look up anyone in the registry">
        Search a name to see every author page it matches: regular pages, numbered namesakes, and the disambiguation bins
        that hold papers dblp hasn’t assigned yet.
      </PageHead>
      <Filters page>
        <SearchBox id="author-q" value={q} onChange={setQ} placeholder="Search a name, e.g. Wei Wang or Jürgen Schmidhuber" autoFocus={!key} />
      </Filters>
      {key ? <AuthorDetail key={key} authorKey={key} go={go} back={() => go("authors", { q: params.q })} /> : null}
      {!key && qd.trim().length >= 2 ? (
        <Card state={search} title={search.data ? `${search.data.length === 30 ? "First 30" : search.data.length} matching author pages` : "Matching author pages"}
              sub="Exact name matches first, then by number of papers. Click a row to open the page.">
          {(d) => (d.length ? (
            <SortableTable rows={d} defaultSort={{ key: "papers", dir: -1 }} onRowClick={(r) => go("authors", { q: qd, key: r.key })} columns={[
              { key: "name", label: "Name", render: (r) => <span className="strong">{r.name}</span> },
              { key: "page_kind", label: "Page", render: (r) => <KindBadge kind={r.page_kind} /> },
              { key: "papers", label: "Papers", num: true, render: (r) => fmt.comma(r.papers) },
              { key: "first_year", label: "Active", render: (r) => (r.first_year ? `${r.first_year}–${r.last_year}` : "—") },
              { key: "affiliation", label: "Affiliation", render: (r) => r.affiliation || <span className="muted">none recorded</span> },
              { key: "namesakes", label: "Pages sharing the name", num: true, render: (r) => fmt.comma(r.namesakes) },
            ]} />
          ) : <EmptyNote>No author page matches “{qd}”.</EmptyNote>)}
        </Card>
      ) : null}
      {!key && qd.trim().length < 2 ? <Callout>Try <button type="button" className="linkish" onClick={() => setQ("Wei Wang")}>Wei Wang</button>{" "}
        (hundreds of namesakes), <button type="button" className="linkish" onClick={() => setQ("Schmidhuber")}>Schmidhuber</button>, or your own name.</Callout> : null}
    </>
  );
}

function AuthorDetail({ authorKey, go, back }) {
  const p = usePalette();
  const d = useApi("authors/detail", { key: authorKey });
  const data = d.data;
  const piv = useMemo(() => data && yearsPivot(data.yearly), [data]);
  const colors = KIND_COLORS(p);
  const yearlyOpt = useMemo(() => piv && barOption(p, { labels: piv.years, stacked: true, fmtY: fmt.comma,
    series: KINDS.filter((k) => piv.kinds.includes(k)).map((k) => ({ name: k, data: piv.get(k), color: colors[k] })) }), [p, piv]);
  if (d.error) return <Callout>Couldn’t load this author page: {d.error}</Callout>;
  if (!data) return <Card state={d} title="Loading author page…" />;
  const { person, stats } = data;
  const isBin = person.page_kind === "disambiguation";
  const links = (person.urls || []).filter((u) => /^https?:/.test(u));
  return (
    <div className="detail">
      <div className="detailhead">
        <button type="button" className="linkish" onClick={back}>&larr; Back to results</button>
        <h2>{person.name}</h2>
        <div className="metaline"><KindBadge kind={person.page_kind} /><code>{person.key}</code>
          {person.names.length > 1 ? <span>also published as {person.names.slice(1).map((n) => `“${n}”`).join(", ")}</span> : null}
        </div>
        {person.affiliations.length ? <div className="metaline">{person.affiliations.map((a, i) => <span key={i} className="affil">{a}</span>)}</div> : null}
        {links.length ? <div className="metaline">{links.map((u) => <a key={u} href={u} target="_blank" rel="noreferrer" className="extlink">{hostOf(u)}</a>)}</div> : null}
      </div>
      {isBin ? (
        <Callout><b>This is a disambiguation bin, not a person.</b> Papers by different people named “{person.name}” sit here until
          dblp’s editors assign them to one of the {fmt.comma(data.namesake_count - 1)} numbered pages below. Everything counted on this page mixes those people.</Callout>
      ) : null}
      <KpiStrip items={[
        { n: fmt.comma(stats.papers), l: "publications (all types)" },
        { n: stats.first_year ? `${stats.first_year}–${stats.last_year}` : "—", l: "years with publications" },
        { n: fmt.comma(stats.coauthors), l: "distinct co-authors (papers with 2–50 authors)" },
        { n: stats.on_3plus ? `${Math.round((100 * stats.first_author) / stats.on_3plus)}% / ${Math.round((100 * stats.last_author) / stats.on_3plus)}%` : "—", l: "first / last author on 3+ author papers" },
        { n: `${stats.pct_with_orcid ?? 0}%`, l: "of their author slots carry an ORCID" },
        { n: fmt.comma(data.namesake_count), l: `author pages named “${person.base_name}”` },
      ]} />
      <div className="grid">
        {isBin ? <BinSplit authorKey={authorKey} go={go} /> : null}
        <Card span2 title="Publications per year" sub="By kind of record.">
          {data.yearly.length ? <Chart option={yearlyOpt} height={240} label="Publications per year" /> : <EmptyNote>No publications resolve to this page.</EmptyNote>}
        </Card>
        <Card title="Most frequent co-authors" sub="Click to open their page.">
          {data.coauthors.length ? (
            <SortableTable rows={data.coauthors} defaultSort={{ key: "papers", dir: -1 }} onRowClick={(r) => go("authors", { key: r.key })} columns={[
              { key: "name", label: "Co-author", render: (r) => <>{r.name} {r.page_kind !== "regular" ? <KindBadge kind={r.page_kind} /> : null}</> },
              { key: "papers", label: "Papers", num: true },
            ]} />
          ) : <EmptyNote>No co-authored papers.</EmptyNote>}
        </Card>
        <Card title="Venues" sub="Journal and conference series. Click to explore one.">
          {data.venues.length ? (
            <SortableTable rows={data.venues} defaultSort={{ key: "papers", dir: -1 }} onRowClick={(r) => go("venues", { sid: r.sid })} columns={[
              { key: "name", label: "Venue" }, { key: "sid", label: "Series", render: (r) => <code>{r.sid}</code> },
              { key: "papers", label: "Papers", num: true },
            ]} />
          ) : <EmptyNote>No journal or conference papers.</EmptyNote>}
        </Card>
        <Card span2 title="Recent publications" sub={`The ${data.papers.length} most recent. Click one to see its record.`}>
          <PaperList rows={data.papers} go={go} showPosition />
        </Card>
        {data.namesakes.length ? (
          <Card span2 title={`Other pages named “${person.base_name}”`} sub={`${fmt.comma(data.namesake_count)} pages share this name; the busiest are listed.`}>
            <SortableTable rows={data.namesakes} defaultSort={{ key: "papers", dir: -1 }} onRowClick={(r) => go("authors", { key: r.key })} columns={[
              { key: "name", label: "Page" }, { key: "page_kind", label: "Kind", render: (r) => <KindBadge kind={r.page_kind} /> },
              { key: "key", label: "Key", render: (r) => <code>{r.key}</code> },
              { key: "papers", label: "Publications", num: true, render: (r) => fmt.comma(r.papers) },
            ]} />
          </Card>
        ) : null}
      </div>
    </div>
  );
}

function hostOf(u) {
  try {
    return new URL(u).hostname.replace(/^www\./, "");
  } catch {
    return u;
  }
}

function PaperList({ rows, go, showPosition }) {
  if (!rows.length) return <EmptyNote>Nothing to show.</EmptyNote>;
  return (
    <ul className="paperlist">
      {rows.map((r) => (
        <li key={r.key}>
          <button type="button" className="paperrow" onClick={() => go("papers", { key: r.key })}>
            <span className="ptitle">{r.title}</span>
            <span className="pmeta">
              <PubKind kind={r.kind} /> {r.year ?? "—"} · {r.venue || <span className="muted">no venue</span>}
              {r.n_authors ? ` · ${r.n_authors} author${r.n_authors === 1 ? "" : "s"}` : ""}
              {showPosition && r.position ? ` · position ${r.position}` : ""}
              {r.n_unidentified ? <span className="flag bad">{r.n_unidentified} unidentified</span> : null}
              {r.has_twin ? <span className="flag">has a twin</span> : null}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

// =============================================================== Venues ====
export function PageVenueExplorer({ params, go }) {
  const [q, setQ, qd] = useQueryParam(params, go, "venues", "q");
  const [kind, setKind] = useState("");
  const sid = params.sid;
  const search = useApi("venues/search", { q: qd.trim(), kind, limit: 40 });
  return (
    <>
      <PageHead eyebrow="Explore · Venues" title="Every journal and conference series">
        A series is dblp’s stable key (<code>conf/cvpr</code>, <code>journals/access</code>), which survives renames that split the venue-name string.
      </PageHead>
      <Filters page>
        <SearchBox id="venue-q" value={q} onChange={setQ} placeholder="Search a venue, e.g. NeurIPS, IEEE Access, conf/kdd" autoFocus={!sid} />
        <Seg label="Kind" value={kind} onChange={setKind} options={[{ v: "", l: "All" }, { v: "conference", l: "Conferences" }, { v: "journal", l: "Journals" }]} />
      </Filters>
      {sid ? <VenueDetail key={sid} sid={sid} go={go} back={() => go("venues", { q: params.q })} /> : (
        <Card state={search} title={qd.trim() ? "Matching series" : "Largest series"} sub="Click a row to open the series.">
          {(d) => (d.length ? (
            <SortableTable rows={d} defaultSort={{ key: "papers", dir: -1 }} onRowClick={(r) => go("venues", { q: qd, sid: r.sid })} columns={[
              { key: "usual_name", label: "Venue", render: (r) => <span className="strong">{r.usual_name}</span> },
              { key: "sid", label: "Series", render: (r) => <code>{r.sid}</code> },
              { key: "kind", label: "Kind", render: (r) => <PubKind kind={r.kind} /> },
              { key: "papers", label: "Papers", num: true, render: (r) => fmt.comma(r.papers) },
              { key: "first_year", label: "Years", render: (r) => `${r.first_year ?? "?"}–${r.last_year ?? "?"}` },
              { key: "name_variants", label: "Name strings", num: true },
              { key: "pct_doi", label: "DOI", num: true, render: (r) => `${r.pct_doi}%` },
              { key: "pct_oa", label: "Open access", num: true, render: (r) => `${r.pct_oa}%` },
            ]} />
          ) : <EmptyNote>No series matches “{qd}”.</EmptyNote>)}
        </Card>
      )}
    </>
  );
}

function VenueDetail({ sid, go, back }) {
  const p = usePalette();
  const d = useApi("venues/detail", { sid });
  const data = d.data;
  const papersOpt = useMemo(() => data && barOption(p, { labels: data.yearly.map((r) => r.year), fmtY: fmt.comma,
    series: [{ name: "Papers", data: data.yearly.map((r) => r.papers), color: data.series.kind === "journal" ? p.s2 : p.s1 }] }), [p, data]);
  const sharesOpt = useMemo(() => data && lineOption(p, { labels: data.yearly.map((r) => r.year), fmtY: fmt.frac, series: [
    { name: "DOI", data: data.yearly.map((r) => r.doi_share), color: p.s1 },
    { name: "Open access", data: data.yearly.map((r) => r.oa_share), color: p.s3 },
    { name: "Any ORCID", data: data.yearly.map((r) => r.orcid_share), color: p.s6 },
    { name: "Unidentified author", data: data.yearly.map((r) => r.unidentified_share), color: p.bad },
    { name: "Has a twin", data: data.yearly.map((r) => r.twin_share), color: p.s4 },
  ] }), [p, data]);
  const teamOpt = useMemo(() => data && lineOption(p, { labels: data.yearly.map((r) => r.year), fmtY: fmt.fixed1,
    series: [{ name: "Mean authors", data: data.yearly.map((r) => r.mean_authors), color: p.s1 }] }), [p, data]);
  if (d.error) return <Callout>Couldn’t load this series: {d.error}</Callout>;
  if (!data) return <Card state={d} title="Loading series…" />;
  const s = data.series;
  return (
    <div className="detail">
      <div className="detailhead">
        <button type="button" className="linkish" onClick={back}>&larr; Back to venues</button>
        <h2>{s.usual_name}</h2>
        <div className="metaline"><PubKind kind={s.kind} /><code>{s.sid}</code>
          {s.name_variants > 1 ? <span>{s.name_variants} different name strings</span> : null}</div>
      </div>
      <KpiStrip items={[
        { n: fmt.comma(s.papers), l: "papers (preprints excluded)" },
        { n: `${s.first_year}–${s.last_year}`, l: `${s.active_years} years with papers` },
        { n: `${Math.round(100 * s.doi_share)}%`, l: "carry a DOI" },
        { n: `${Math.round(100 * s.oa_share)}%`, l: "flagged open access" },
      ]} />
      <div className="grid">
        <Card span2 title="Papers per year"><Chart option={papersOpt} height={230} label="Papers per year" /></Card>
        <Card title="Metadata over time" sub="Share of the series’ papers each year."><Chart option={sharesOpt} label="Metadata shares" /></Card>
        <Card title="Mean authors per paper"><Chart option={teamOpt} label="Team size" /></Card>
        <Card title="Name strings" sub="How the venue name was written over time; all belong to this one series.">
          <SortableTable rows={data.names} defaultSort={{ key: "papers", dir: -1 }} columns={[
            { key: "name", label: "Name string" }, { key: "papers", label: "Papers", num: true, render: (r) => fmt.comma(r.papers) },
            { key: "first_year", label: "Years", render: (r) => `${r.first_year}–${r.last_year}` },
          ]} />
        </Card>
        <Card title="Most frequent authors" sub="Disambiguation bins excluded. Click to open.">
          <SortableTable rows={data.top_authors} defaultSort={{ key: "papers", dir: -1 }} onRowClick={(r) => go("authors", { key: r.key })} columns={[
            { key: "name", label: "Author" }, { key: "papers", label: "Papers", num: true },
          ]} />
        </Card>
        <Card span2 title="Latest papers">
          <PaperList rows={data.recent.map((r) => ({ ...r, kind: s.kind, venue: s.usual_name }))} go={go} />
        </Card>
      </div>
    </div>
  );
}

// =============================================================== Papers ====
export function PagePapers({ params, go }) {
  const [q, setQ, qd] = useQueryParam(params, go, "papers", "q");
  const [kind, setKind] = useState("");
  const [from, setFrom] = useState(1970);
  const fromD = useDebounced(from, 500);
  const key = params.key;
  const ready = qd.trim().length >= 3;
  const search = useApi(ready ? "papers/search" : null, { q: qd.trim(), kind, from: fromD > 1970 ? fromD : undefined, limit: 40 });
  return (
    <>
      <PageHead eyebrow="Explore · Papers" title="Find a publication and read its record">
        Every word must appear in the title. Opening a paper shows its authors resolved through the registry and the record as dblp stores it.
      </PageHead>
      <Filters page>
        <SearchBox id="paper-q" value={q} onChange={setQ} placeholder="Search titles, e.g. attention is all you need" autoFocus={!key} />
        <Seg label="Kind" value={kind} onChange={setKind} options={[{ v: "", l: "All" }, { v: "journal", l: "Journal" }, { v: "conference", l: "Conference" }, { v: "preprint", l: "Preprint" }]} />
        <FilterLabel htmlFor="paper-from">From</FilterLabel>
        <NumberInput id="paper-from" label="From year" value={from} min={1936} max={2026} onChange={setFrom} />
      </Filters>
      {key ? <PaperDetail key={key} paperKey={key} go={go} back={() => go("papers", { q: params.q })} /> : null}
      {!key && ready ? (
        <Card state={search} title="Matching publications" sub="Newest first, up to 40.">
          {(d) => (d.length ? <PaperList rows={d} go={go} /> : <EmptyNote>No title contains all of “{qd}”.</EmptyNote>)}
        </Card>
      ) : null}
      {!key && !ready ? <Callout>Try <button type="button" className="linkish" onClick={() => setQ("DeepSeek-R1")}>DeepSeek-R1</button>{" "}
        (a Nature paper with 194 authors) or <button type="button" className="linkish" onClick={() => setQ("attention is all you need")}>attention is all you need</button>.</Callout> : null}
    </>
  );
}

const XML_ORDER = ["title", "pages", "year", "volume", "journal", "booktitle", "number", "publisher", "school", "series"];

/** Rebuild an XML view of the record from the parsed fields (the parser keeps 28 of dblp's fields). */
function RecordXml({ record }) {
  const lines = [];
  const attrs = [["mdate", record.mdate], ["key", record.key], ["publtype", record.publtype]].filter(([, v]) => v);
  lines.push({ open: record.type, attrs });
  (record.authors || []).forEach((a, i) => lines.push({ tag: "author", text: a, attrs: record.author_orcids?.[i] ? [["orcid", record.author_orcids[i]]] : [] }));
  (record.editors || []).forEach((e) => lines.push({ tag: "editor", text: e }));
  XML_ORDER.forEach((f) => { if (record[f] !== null && record[f] !== undefined) lines.push({ tag: f, text: String(record[f]) }); });
  (record.isbn || []).forEach((v) => lines.push({ tag: "isbn", text: v }));
  (record.ee || []).forEach((v) => lines.push({ tag: "ee", text: v }));
  if (record.crossref) lines.push({ tag: "crossref", text: record.crossref });
  (record.urls || []).forEach((v) => lines.push({ tag: "url", text: v }));
  (record.notes || []).forEach((n) => {
    const i = n.indexOf(": ");
    const type = i > 0 ? n.slice(0, i) : "note";
    lines.push({ tag: "note", text: i > 0 ? n.slice(i + 2) : n, attrs: type !== "note" ? [["type", type]] : [] });
  });
  if (record.has_oa) lines.push({ comment: 'one of the links above is marked type="oa" (open access)' });
  lines.push({ close: record.type });
  const A = ({ attrs: at }) => (at || []).map(([k, v]) => <span key={k}> <span className="xa">{k}</span>=<span className="xv">"{v}"</span></span>);
  return (
    <pre className="xml" aria-label="Record as XML">
      {lines.map((l, i) => {
        if (l.open) return <div key={i}><span className="xt">&lt;{l.open}</span><A attrs={l.attrs} /><span className="xt">&gt;</span></div>;
        if (l.close) return <div key={i}><span className="xt">&lt;/{l.close}&gt;</span></div>;
        if (l.comment) return <div key={i} className="xc">{"  "}&lt;!-- {l.comment} --&gt;</div>;
        return <div key={i}>{"  "}<span className="xt">&lt;{l.tag}</span><A attrs={l.attrs} /><span className="xt">&gt;</span>{l.text}<span className="xt">&lt;/{l.tag}&gt;</span></div>;
      })}
    </pre>
  );
}

function PaperDetail({ paperKey, go, back }) {
  const d = useApi("papers/detail", { key: paperKey });
  const data = d.data;
  const [showAll, setShowAll] = useState(false);
  if (d.error) return <Callout>Couldn’t load this publication: {d.error}</Callout>;
  if (!data) return <Card state={d} title="Loading publication…" />;
  const { paper, record, authors, twins } = data;
  const counts = authors.reduce((m, a) => ({ ...m, [a.page_kind]: (m[a.page_kind] || 0) + 1 }), {});
  const shown = showAll ? authors : authors.slice(0, 40);
  const doi = (record.ee || []).find((u) => u.includes("doi.org/"));
  return (
    <div className="detail">
      <div className="detailhead">
        <button type="button" className="linkish" onClick={back}>&larr; Back to results</button>
        <h2>{record.title}</h2>
        <div className="metaline">
          <PubKind kind={paper.kind} /> {record.year}
          {paper.venue ? <> · {paper.sid && /^(conf|journals)\//.test(paper.sid) && !paper.is_preprint
            ? <button type="button" className="linkish" onClick={() => go("venues", { sid: paper.sid })}>{paper.venue}</button>
            : paper.venue}</> : null}
          <code>{record.key}</code>
        </div>
        <div className="metaline">
          {doi ? <a className="extlink" href={doi} target="_blank" rel="noreferrer">DOI</a> : <span className="flag">no DOI</span>}
          <a className="extlink" href={`https://dblp.org/rec/${record.key}.html`} target="_blank" rel="noreferrer">dblp</a>
          {record.has_oa ? <span className="flag good">open access</span> : null}
          {record.publtype ? <span className="flag">{record.publtype}</span> : null}
        </div>
      </div>
      {!authors.length ? <Callout><b>This record has no authors.</b>{record.publtype === "withdrawn" ? " dblp removes the author list when a paper is withdrawn." : ""}</Callout> : null}
      {counts.disambiguation ? (
        <Callout><b>{counts.disambiguation} of {authors.length} authors</b> resolve to a disambiguation bin: dblp knows the name, not the person.
          {counts.unresolved ? ` ${counts.unresolved} more match no author page at all.` : ""}</Callout>
      ) : null}
      <div className="grid">
        <Card span2 title={`Authors (${authors.length})`} sub="Resolved through the author-page registry. Click one to open their page.">
          {authors.length ? (
            <>
              <ol className="authorlist">
                {shown.map((a) => (
                  <li key={a.position}>
                    {a.key ? <button type="button" className="linkish" onClick={() => go("authors", { key: a.key })}>{a.name}</button> : <span>{a.name}</span>}
                    {a.page_kind !== "regular" ? <KindBadge kind={a.page_kind} /> : null}
                    {a.orcid ? <a className="orcid" href={`https://orcid.org/${a.orcid}`} target="_blank" rel="noreferrer" title={`ORCID ${a.orcid}`}>iD</a> : null}
                  </li>
                ))}
              </ol>
              {authors.length > 40 && !showAll ? <button type="button" className="btn" onClick={() => setShowAll(true)}>Show all {authors.length}</button> : null}
            </>
          ) : <EmptyNote>No authors on this record.</EmptyNote>}
        </Card>
        {twins.length ? (
          <Card span2 title="Same title, separate records" sub="dblp does not link a preprint to its published version; these share the normalized title.">
            <PaperList rows={twins} go={go} />
          </Card>
        ) : null}
        <Card span2 title="The record" sub="Rebuilt from the parsed fields. The parser keeps 28 fields, so tags like <month> or <stream> in the original XML are not shown.">
          <div className="tablewrap"><RecordXml record={record} /></div>
        </Card>
      </div>
    </div>
  );
}
