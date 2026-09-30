import { useMemo, useState } from "react";
import { useApi, useDebounced, useStatic } from "./api.js";
import { Chart, usePalette, lineOption, barOption, hbarOption, loglogOption, growthOption, treemapOption, scatterOption, densityOption, boxplotOption, fmt } from "./charts.jsx";
import {
  KpiStrip, PageHead, Card, Filters, FilterLabel, Chip, ChipGroup, Seg, RangeSlider, NumberInput,
  SortableTable, Heatmap, Callout, EmptyNote,
} from "./components.jsx";

const cap = (s) => s[0].toUpperCase() + s.slice(1);
const pct = (v) => (v == null ? "—" : `${v}%`);

/** Last complete year of the dump (from the server), used as the default end of time ranges. */
function useLastYear(status) {
  return Number(status?.meta?.last_full_year) || 2025;
}

// ============================================================ Overview ====
export function PageOverview({ go }) {
  const p = usePalette();
  const ov = useApi("overview");
  const growth = useApi("publishing/growth");
  const hom = useApi("identity/homonyms", { top: 10 });
  const net = useApi("network");
  const growthOpt = useMemo(() => growth.data && lineOption(p, {
    labels: growth.data.map((r) => r.year), log: true, fmtY: fmt.comma, series: [
      { name: "Conference", data: growth.data.map((r) => r.conference), color: p.s1 },
      { name: "Journal", data: growth.data.map((r) => r.journal), color: p.s2 },
      { name: "Preprint", data: growth.data.map((r) => r.preprint), color: p.s3 },
    ] }), [p, growth.data]);
  const homOpt = useMemo(() => hom.data && hbarOption(p, {
    labels: hom.data.map((r) => r.base_name), fmtX: fmt.comma, labelWidth: 90,
    series: [{ name: "People", data: hom.data.map((r) => r.distinct_people), color: p.s1 }] }), [p, hom.data]);
  const netData = net.data?.available ? net.data : null;
  const netOpt = useMemo(() => netData && lineOption(p, {
    labels: netData.growth.map((r) => r.up_to), fmtY: fmt.pct,
    series: [{ name: "Share in giant component", data: netData.growth.map((r) => r.giant_pct), color: p.s3 }] }), [p, netData]);
  return (
    <>
      <PageHead eyebrow="dblp · live" title="Reading dblp">
        dblp looks like a list of CS papers. It’s really two datasets braided together: publications, and a
        registry of millions of author pages deciding who wrote them. Every number here is queried from the dump.
      </PageHead>
      <KpiStrip items={ov.data?.kpis} loading={!ov.data} />
      {ov.error ? <Callout>Couldn’t load the headline numbers: {ov.error}</Callout> : null}
      <div className="grid">
        <Card span2 state={growth} height={300} title="Papers per year, by kind"
              sub="Log scale. Conferences overtook journals in 1990; journals retook the lead in 2020 as mega-journals grew.">
          {() => <Chart option={growthOpt} height={300} label="Papers per year by kind" />}
        </Card>
        <Card state={hom} height={300} title="One name, hundreds of people"
              sub="Distinct author pages sharing a base name. Click a bar to see them.">
          {() => <Chart option={homOpt} height={300} label="Most shared names"
                        onClick={(e) => go("authors", { q: e.name })} />}
        </Card>
        <Card state={net} job={net.data?.source} height={300} title="The co-authorship network"
              sub="Share of connected authors inside the one giant, small-world component.">
          {() => (netData ? <Chart option={netOpt} height={300} label="Network growth" />
            : <EmptyNote>{net.data?.reason}</EmptyNote>)}
        </Card>
      </div>
    </>
  );
}

// ========================================================== Publishing ====
export function PagePublishing({ status }) {
  const p = usePalette();
  const last = useLastYear(status);
  const [range, setRange] = useState([1970, last]);
  const r = useDebounced(range);
  const [kinds, setKinds] = useState({ conference: true, journal: true, preprint: true });
  const [rateCfg, setRateCfg] = useState({ from: 1990, split: 2005, to: 2023 });
  const rc = useDebounced(rateCfg, 600);
  const kindColor = { conference: p.s1, journal: p.s2, preprint: p.s3 };

  const growth = useApi("publishing/growth", { from: r[0], to: r[1] });
  const teams = useApi("publishing/teams", { from: r[0], to: r[1] });
  const boxes = useApi("publishing/team-boxes", { from: r[0], to: r[1], step: 5 });
  const meta = useApi("publishing/metadata", { from: Math.max(2000, r[0]), to: r[1] });
  const rates = useApi("publishing/growth-rates", rc);

  const growthOpt = useMemo(() => growth.data && lineOption(p, {
    labels: growth.data.map((x) => x.year), log: true, fmtY: fmt.comma,
    series: Object.keys(kinds).filter((k) => kinds[k]).map((k) => ({ name: cap(k), data: growth.data.map((x) => x[k]), color: kindColor[k] })),
  }), [p, growth.data, kinds]);
  const teamOpt = useMemo(() => teams.data && lineOption(p, { labels: teams.data.map((x) => x.year), fmtY: fmt.fixed1,
    series: [{ name: "Mean authors", data: teams.data.map((x) => x.mean_authors), color: p.s1 }] }), [p, teams.data]);
  const soloOpt = useMemo(() => teams.data && lineOption(p, { labels: teams.data.map((x) => x.year), fmtY: fmt.frac,
    series: [
      { name: "Single author", data: teams.data.map((x) => x.single_author_share), color: p.s2 },
      { name: "10+ authors", data: teams.data.map((x) => x.share_10plus), color: p.s6 },
    ] }), [p, teams.data]);
  const metaOpt = useMemo(() => meta.data && lineOption(p, { labels: meta.data.map((x) => x.year), fmtY: fmt.frac, series: [
    { name: "Has an ORCID", data: meta.data.map((x) => x.with_orcid), color: p.s1 },
    { name: "Unidentified author", data: meta.data.map((x) => x.with_unidentified_author), color: p.bad },
    { name: "Preprint/published twin", data: meta.data.map((x) => x.with_title_twin), color: p.s4 },
    { name: "Open access", data: meta.data.map((x) => x.open_access), color: p.s3 },
  ] }), [p, meta.data]);
  const boxOpt = useMemo(() => boxes.data && boxplotOption(p, { fmtY: fmt.fixed1,
    labels: boxes.data.map((x) => (x.period_end > x.period_start ? `${x.period_start}–${String(x.period_end).slice(2)}` : `${x.period_start}`)),
    boxes: boxes.data.map((x) => [x.p5, x.q1, x.median, x.q3, x.p95]),
    means: boxes.data.map((x) => x.mean) }), [p, boxes.data]);
  const ratesRows = rates.data?.rows.filter((x) => x.period !== `${rates.data.from}-${rates.data.to}`);
  const ratesOpt = useMemo(() => ratesRows && growthOption(p, ratesRows), [p, rates.data]);
  const full = rates.data?.rows.find((x) => x.kind === "all" && x.period === `${rates.data.from}-${rates.data.to}`);

  return (
    <>
      <PageHead eyebrow="01 · Fifty years of publishing" title="Growth, teams, and where the papers go">
        Conferences, journals and preprints tell very different stories. The year range applies to every chart below.
      </PageHead>
      <Filters page>
        <FilterLabel htmlFor="yr-from">Years</FilterLabel>
        <RangeSlider id="yr-from" label="First year" min={1970} max={range[1]} value={range[0]} onChange={(v) => setRange([v, range[1]])} />
        <span className="muted">to</span>
        <RangeSlider id="yr-to" label="Last year" min={range[0]} max={last + 1} value={range[1]} onChange={(v) => setRange([range[0], v])} />
        {range[1] > last ? <span className="hint">{last + 1} is still being indexed</span> : null}
        <FilterLabel>Kind</FilterLabel>
        <ChipGroup>
          {Object.keys(kinds).map((k) => <Chip key={k} label={cap(k)} on={kinds[k]} color={kindColor[k]} onClick={() => setKinds({ ...kinds, [k]: !kinds[k] })} />)}
        </ChipGroup>
      </Filters>
      <div className="grid">
        <Card span2 state={growth} height={300} title="Papers per year, by kind" sub="Log scale: a straight line is steady percentage growth.">
          {() => <Chart option={growthOpt} height={300} label="Papers per year" />}
        </Card>
        <Card state={teams} title="Mean authors per paper" sub="Journal and conference papers, preprints excluded.">
          {() => <Chart option={teamOpt} label="Mean authors per paper" />}
        </Card>
        <Card state={teams} title="Solo papers vanished; big teams grew" sub="Share of papers with one author, and with ten or more.">
          {() => <Chart option={soloOpt} label="Team size shares" />}
        </Card>
        <Card span2 state={boxes} height={300} title="Authors per paper: the whole distribution, not just its mean"
              sub={boxes.data?.length > 1
                ? `Five-year periods. Box = middle half of papers, whiskers = 5th to 95th percentile, dot = mean. From ${boxes.data[0].period_start}s to ${boxes.data.at(-1).period_start}s the median went ${boxes.data[0].median} → ${boxes.data.at(-1).median}, the 95th percentile ${boxes.data[0].p95} → ${boxes.data.at(-1).p95}.`
                : "Five-year periods. Box = middle half of papers, whiskers = 5th to 95th percentile, dot = mean."}>
          {() => <Chart option={boxOpt} height={300} label="Authors per paper by period" />}
        </Card>
        <Card span2 state={meta} height={290} title="Four metadata trends"
              sub="Share of journal, conference and preprint records per year (from 2000). Recent twins are undercounted: many preprints aren’t published yet.">
          {() => <Chart option={metaOpt} height={290} label="Metadata trends" />}
        </Card>
        <Card span2 state={rates} height={270} title="Annual growth rate"
              sub={full ? `All kinds, ${rates.data.from}–${rates.data.to}: ${full.rate}%/yr (95% CI ${full.lo}–${full.hi}%), doubling every ${full.doubling} years. Log-linear OLS; the interval is optimistic because years aren’t independent.` : "Log-linear OLS on yearly counts."}
              controls={<>
                <FilterLabel htmlFor="rate-from">Fit</FilterLabel>
                <NumberInput id="rate-from" label="Fit from" value={rateCfg.from} min={1970} max={rateCfg.split - 3} onChange={(v) => setRateCfg({ ...rateCfg, from: v })} />
                <span className="muted">split after</span>
                <NumberInput id="rate-split" label="Split after year" value={rateCfg.split} min={rateCfg.from + 3} max={rateCfg.to - 3} onChange={(v) => setRateCfg({ ...rateCfg, split: v })} />
                <span className="muted">to</span>
                <NumberInput id="rate-to" label="Fit to" value={rateCfg.to} min={rateCfg.split + 3} max={last} onChange={(v) => setRateCfg({ ...rateCfg, to: v })} />
              </>}>
          {() => <Chart option={ratesOpt} height={270} label="Growth rates" />}
        </Card>
      </div>
    </>
  );
}

// ============================================================ Identity ====
export function PageIdentity({ go }) {
  const p = usePalette();
  const [homN, setHomN] = useState(10);
  const top = useDebounced(homN, 250);
  const [since, setSince] = useState(2015);
  const [minPapers, setMinPapers] = useState(2000);
  const minP = useDebounced(minPapers, 600);

  const hom = useApi("identity/homonyms", { top });
  const aff = useApi("identity/affiliation");
  const newc = useApi("identity/newcomers");
  const cohorts = useApi("identity/cohorts");
  const pos = useApi("identity/positions");
  const unid = useApi("identity/unidentified-by-position", { since });
  const alpha = useApi("identity/alphabetical", { min_papers: minP });

  const homOpt = useMemo(() => hom.data && hbarOption(p, { labels: hom.data.map((x) => x.base_name), fmtX: fmt.comma, labelWidth: 90,
    series: [{ name: "People", data: hom.data.map((x) => x.distinct_people), color: p.s1 }] }), [p, hom.data]);
  const affOpt = useMemo(() => aff.data?.regular && hbarOption(p, {
    labels: ["Has affiliation", "Links ORCID", "Links Wikidata", "Name variants"], fmtX: fmt.frac, labelWidth: 110,
    series: [
      { name: "Regular pages", color: p.s1, data: ["affiliation", "orcid", "wikidata", "name_variants"].map((k) => aff.data.regular[k]) },
      { name: "Numbered namesakes", color: p.s4, data: ["affiliation", "orcid", "wikidata", "name_variants"].map((k) => aff.data.numbered?.[k]) },
    ] }), [p, aff.data]);
  const newOpt = useMemo(() => newc.data && barOption(p, { labels: newc.data.map((x) => x.year), fmtY: fmt.comma,
    series: [{ name: "First-time authors", data: newc.data.map((x) => x.new_authors), color: p.s3 }] }), [p, newc.data]);
  const cohortOpt = useMemo(() => cohorts.data && lineOption(p, { labels: cohorts.data.map((x) => x.cohort), fmtY: fmt.pct, series: [
    { name: "Still publishing 5 y later", data: cohorts.data.map((x) => x.pct_5y), color: p.s1 },
    { name: "10 y later", data: cohorts.data.map((x) => x.pct_10y), color: p.s2 },
    { name: "20 y later", data: cohorts.data.map((x) => x.pct_20y), color: p.s6 },
  ] }), [p, cohorts.data]);
  const posOpt = useMemo(() => pos.data && hbarOption(p, { labels: pos.data.map((x) => `${x.author_total_papers} papers`), stacked: true, max: 100, fmtX: fmt.pct, labelWidth: 100, series: [
    { name: "First author", color: p.s1, data: pos.data.map((x) => x.pct_first) },
    { name: "Middle", color: p.muted, data: pos.data.map((x) => x.pct_middle) },
    { name: "Last author", color: p.s2, data: pos.data.map((x) => x.pct_last) },
  ] }), [p, pos.data]);
  const unidOpt = useMemo(() => unid.data && barOption(p, { labels: unid.data.map((x) => cap(x.author_position)), fmtY: (v) => `${v}%`, series: [
    { name: "On a disambiguation bin", data: unid.data.map((x) => x.pct_unidentified), color: p.bad },
    { name: "Carries an ORCID", data: unid.data.map((x) => x.pct_with_orcid), color: p.s1 },
  ] }), [p, unid.data]);
  const alphaRows = alpha.data && [...alpha.data.most, ...[...alpha.data.least].reverse()];
  const alphaOpt = useMemo(() => alphaRows && hbarOption(p, { labels: alphaRows.map((x) => x.usual_name), fmtX: fmt.pct, labelWidth: 210, series: [
    { name: "Actually alphabetical", color: p.s1, data: alphaRows.map((x) => x.pct_alphabetical) },
    { name: "Expected by chance", color: p.muted, data: alphaRows.map((x) => x.pct_by_chance) },
  ] }), [p, alpha.data]);

  return (
    <>
      <PageHead eyebrow="02 · Author identity" title="One name, hundreds of people">
        Papers carry no author ID, only a name string, resolved through the author-page registry. It fails in three ways:
        name variants, numbered namesakes, and unassigned disambiguation bins.
      </PageHead>
      <div className="grid">
        <Card state={hom} height={homN > 10 ? 340 : 280} title="Most-shared names" sub="Distinct author pages per base name. Click a bar to see them."
              controls={<><FilterLabel htmlFor="hom-n">Show top</FilterLabel><RangeSlider id="hom-n" label="Number of names" min={5} max={25} value={homN} onChange={setHomN} /></>}>
          {() => <Chart option={homOpt} height={homN > 10 ? 340 : 280} label="Most shared names" onClick={(e) => go("authors", { q: e.name })} />}
        </Card>
        <Card state={aff} height={280} title="Affiliations exist to tell people apart" sub="Numbered namesakes almost always carry an affiliation; everyone else almost never does.">
          {() => <Chart option={affOpt} height={280} label="Affiliation effect" />}
        </Card>
        <Card state={newc} title="New authors per year" sub="People whose first journal or conference paper appeared that year.">
          {() => <Chart option={newOpt} label="New authors per year" />}
        </Card>
        <Card state={cohorts} title="Most authors do not stay" sub="Share of each starting cohort still publishing k years later. Horizons past the data end are left out.">
          {() => <Chart option={cohortOpt} label="Cohort survival" />}
        </Card>
        <Card span2 state={pos} height={240} title="Who signs where" sub="Position on papers with 3+ authors, by the author’s career total. Prolific authors sign last.">
          {() => <Chart option={posOpt} height={240} label="Author position" />}
        </Card>
        <Card span2 state={unid} height={250} title="Who goes unidentified, by author position"
              sub="Author slots on 3+ author papers: share landing on a disambiguation bin, and share with an ORCID, by position."
              controls={<><FilterLabel>Papers since</FilterLabel><Seg label="Since" value={since} onChange={setSince}
                options={[2000, 2010, 2015, 2020].map((y) => ({ v: y, l: String(y) }))} /></>}>
          {() => <Chart option={unidOpt} height={250} label="Unidentified by position" />}
        </Card>
        <Card span2 state={alpha} height={440} title="Alphabetical author order"
              sub="Theory venues list authors alphabetically far above chance; applied venues sit at chance. Series with enough papers since 2000 (2–10 authors, distinct surnames)."
              controls={<><FilterLabel htmlFor="alpha-min">Min. papers</FilterLabel>
                <NumberInput id="alpha-min" label="Minimum papers per series" value={minPapers} min={100} max={100000} step={100} onChange={setMinPapers} width={90} /></>}>
          {() => (alphaRows.length ? <Chart option={alphaOpt} height={Math.max(200, 26 * alphaRows.length + 60)} label="Alphabetical order by venue" />
            : <EmptyNote>No series has {fmt.comma(minPapers)} papers since 2000. Lower the minimum.</EmptyNote>)}
        </Card>
      </div>
    </>
  );
}

// ============================================================= Network ====
const DATASET = "/downloads/dblp-coauthor/";
const DATASHEETS = {
  network: { label: "About the graph", title: "How the graph was built",
             lede: "What makes an edge, what was left out and why, and how the files are laid out." },
  centrality: { label: "About the centrality table", title: "How centrality was measured",
                lede: "Which measures are exact and which are estimated, how each was checked, and who comes out on top." },
};

/**
 * A published datasheet, shown as a page of the site. The content is written by the publish step
 * (rendered from its Markdown, with author names escaped there), so the page and the download are
 * always the same document.
 */
export function PageDatasheet({ params, go }) {
  const about = DATASHEETS[params?.about] ? params.about : "network";
  const sheet = DATASHEETS[about];
  const doc = useStatic(`${DATASET}datasheet-${about}.html`, "text");
  const other = about === "network" ? "centrality" : "network";
  const back = () => {
    go("network", {});
    setTimeout(() => document.getElementById("download")?.scrollIntoView({ behavior: "smooth" }), 80);
  };
  return (
    <>
      <PageHead eyebrow="03 · The co-authorship network · datasheet" title={sheet.title}>{sheet.lede}</PageHead>
      <div className="datasheetnav">
        <button type="button" className="linkbtn" onClick={back}>← Back to the download</button>
        <button type="button" className="linkbtn" onClick={() => go("datasheet", { about: other })}>
          {DATASHEETS[other].label} →
        </button>
      </div>
      <section className="card datasheet">
        {doc.missing ? <EmptyNote>This datasheet hasn’t been published on this server yet.</EmptyNote>
          : doc.error ? <div className="cardmsg error" role="alert"><b>Couldn’t load the datasheet.</b> {doc.error}</div>
          : !doc.data ? <div className="skel" style={{ height: 420 }} />
          // written by the publish step from this project's own Markdown; names are escaped there
          : <article className="prose" dangerouslySetInnerHTML={{ __html: doc.data }} />}
      </section>
    </>
  );
}

function bytes(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  if (n >= 1e3) return `${Math.round(n / 1e3)} kB`;
  return `${n} B`;
}

/**
 * The whole graph as files, listed from the manifest the publish step writes - so the page shows what
 * is actually on disk, sizes and checksums included, and cannot drift from it. Served by nginx, so
 * large downloads resume.
 */
function DatasetDownload({ go }) {
  const dl = useStatic(`${DATASET}manifest.json`);
  const m = dl.data;
  const data = m?.files.filter((f) => f.kind === "data") || [];
  const docs = m?.files.filter((f) => f.kind === "doc") || [];
  const s = m?.summary || {};
  let body;
  if (dl.missing) {
    body = <EmptyNote>The dataset hasn’t been published on this server yet.</EmptyNote>;
  } else if (dl.error) {
    body = <div className="cardmsg error" role="alert"><b>Couldn’t load the file list.</b> {dl.error}</div>;
  } else if (!m) {
    body = <div className="skel" style={{ height: 200, marginTop: 10 }} />;
  } else {
    body = (
      <>
        <p className="dlsum">
          {fmt.comma(s.authors_with_a_coauthor)} authors and {fmt.comma(s.coauthorships)} co-authorships from the
          dblp snapshot of {m.dump?.latest_mdate} — every record type, not a sample. {m.format}.
          License: {m.license}.
        </p>
        <div className="tablewrap">
          <table className="data dltable">
            <thead><tr><th>File</th><th>What it is</th><th style={{ textAlign: "right" }}>Size</th></tr></thead>
            <tbody>
              {data.map((f) => (
                <tr key={f.name}>
                  <td><a className="mono" href={DATASET + f.name} download>{f.name}</a></td>
                  <td>{f.description}</td>
                  <td className="num">{bytes(f.bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="dldocs">
          <span>Read first:</span>
          {docs.map((f) => {
            const about = f.name.replace(/^datasheet-|\.html$/g, "");
            return (
              <a key={f.name} href={`#datasheet?about=${about}`}
                 onClick={(e) => { e.preventDefault(); go("datasheet", { about }); }}>
                {DATASHEETS[about]?.label || f.name}
              </a>
            );
          })}
        </div>
      </>
    );
  }
  return (
    <Card id="download" span2 title="Download the network"
          sub={m ? `${data.length} files, ${bytes(m.total_bytes)} in total · published ${m.published_at?.slice(0, 10)}` : "The graph, the node list, venue communities and centrality for every author."}>
      {body}
    </Card>
  );
}

export function PageNetwork({ go }) {
  const p = usePalette();
  const net = useApi("network");
  const d = net.data?.available ? net.data : null;
  const growthOpt = useMemo(() => d && lineOption(p, { labels: d.growth.map((r) => r.up_to), fmtY: fmt.pct,
    series: [{ name: "Share in giant component", data: d.growth.map((r) => r.giant_pct), color: p.s3 }] }), [p, d]);
  const degOpt = useMemo(() => d && lineOption(p, { labels: d.growth.map((r) => r.up_to), fmtY: fmt.fixed1,
    series: [{ name: "Mean co-authors", data: d.growth.map((r) => r.mean_degree), color: p.s1 }] }), [p, d]);
  const distRows = d?.distances.filter((r) => r.hops <= 10) || [];
  const distOpt = useMemo(() => d && barOption(p, { labels: distRows.map((r) => `${r.hops}`), fmtY: fmt.pct,
    series: [{ name: "Share of pairs", data: distRows.map((r) => r.pct_of_pairs), color: p.s1 }] }), [p, d]);
  const clean = d?.overview["without disambiguation pages"];
  const all = d?.overview["all author pages"];
  const s = d?.structure || {};
  const c = d?.communities || {};
  return (
    <>
      <PageHead eyebrow="03 · The co-authorship network" title="A small world, built from author pairs">
        Two authors are linked if they share a paper with 2–50 authors. Building and measuring this graph takes minutes
        of compute, so this page shows the latest run of the network job rather than a per-request query.{" "}
        <button type="button" className="linkbtn"
                onClick={() => document.getElementById("download")?.scrollIntoView({ behavior: "smooth" })}>
          The whole graph can be downloaded.
        </button>
      </PageHead>
      {net.data && !d ? <Callout>{net.data.reason}</Callout> : null}
      <KpiStrip loading={!d} items={d && [
        { n: `${clean?.giant_pct}%`, l: "of connected authors sit in one giant component" },
        { n: `${clean?.mean_degree}`, l: `mean distinct co-authors (max ${clean?.max_degree?.toLocaleString()})` },
        { n: `${d.distance_summary?.mean ?? "—"}`, l: `average steps between two authors; 90% within ${d.distance_summary?.p90 ?? "—"}` },
        { n: `${s["average local clustering"]}`, l: `average local clustering (random graph: ${s["clustering expected in a random graph"]})` },
        { n: `k = ${s["max k-core"]}`, l: `densest core: ${s["authors in the max k-core"]} authors` },
        { n: `${c.communities?.toLocaleString()}`, l: `Leiden communities, modularity ${c.modularity}` },
        { n: `${Math.round(100 - (100 * clean?.edges) / all?.edges)}%`, l: "of all co-author pairs disappear once disambiguation bins are removed" },
      ]} />
      <div className="grid">
        <Card state={net} job={net.data?.source} title="How the network grew" sub="Share of connected authors inside the largest component.">
          {() => d && <Chart option={growthOpt} label="Giant component share" />}
        </Card>
        <Card state={net} job={net.data?.source} title="Mean distinct co-authors" sub="Among authors with at least one co-author by that year.">
          {() => d && <Chart option={degOpt} label="Mean degree" />}
        </Card>
        <Card span2 state={net} job={net.data?.source} height={250} title="Six degrees of separation"
              sub={d?.distance_summary ? `Shortest co-authorship paths from 20 random authors. Mean ${d.distance_summary.mean}, median ${d.distance_summary.median}, 90% within ${d.distance_summary.p90}.` : ""}>
          {() => d && <Chart option={distOpt} height={250} label="Distance distribution" />}
        </Card>
        <Card span2 state={net} job={net.data?.source} title="The largest communities" sub="Leiden detection. Labelled by what their members publish in; click a venue to explore it.">
          {() => d && (
            <SortableTable defaultSort={{ key: "authors", dir: -1 }} rows={d.largest_communities} columns={[
              { key: "authors", label: "Authors", num: true, render: (r) => fmt.comma(r.authors) },
              { key: "venues", label: "Top venues", render: (r) => r.venues.map((v, i) => (
                <button key={i} type="button" className="venuepill" onClick={() => go("venues", { q: v.name })}>
                  {v.name} <span className="muted">{fmt.comma(v.papers)}</span>
                </button>)) },
            ]} />
          )}
        </Card>
        <Card state={net} job={net.data?.source} title="Most-connected authors" sub="Distinct co-authors, disambiguation bins removed. Click to open.">
          {() => d && (
            <SortableTable defaultSort={{ key: "coauthors", dir: -1 }} rows={d.most_connected}
                           onRowClick={(r) => go("authors", { q: r.author })}
                           columns={[{ key: "author", label: "Author" }, { key: "coauthors", label: "Co-authors", num: true, render: (r) => fmt.comma(r.coauthors) }]} />
          )}
        </Card>
        <Card state={net} job={net.data?.source} title="How often pairs work together" sub="Papers written together per co-author pair.">
          {() => d && (
            <SortableTable defaultSort={{ key: "pairs", dir: -1 }} rows={d.edge_weights}
                           columns={[{ key: "papers_together", label: "Papers together" },
                                     { key: "pairs", label: "Pairs", num: true, render: (r) => fmt.comma(r.pairs) },
                                     { key: "pct", label: "Share", num: true, render: (r) => `${r.pct}%` }]} />
          )}
        </Card>
        <DatasetDownload go={go} />
      </div>
    </>
  );
}

// ============================================================== Titles ====
const PRESETS = ["llm", "deep", "neural", "transformer", "graph", "federated", "diffusion", "cloud", "iot", "blockchain", "quantum", "privacy", "explainable"];
const TERM_LABEL = { llm: "LLM", iot: "IoT" };
const termLabel = (t) => TERM_LABEL[t] || t;

export function PageTitles({ status }) {
  const p = usePalette();
  const last = useLastYear(status);
  const palette = [p.s1, p.s2, p.s3, p.s4, p.s5, p.s6, p.bad, p.good];
  const [active, setActive] = useState(["llm", "deep", "neural", "transformer"]);
  const [custom, setCustom] = useState("");
  const [from, setFrom] = useState(2000);
  const fromD = useDebounced(from, 400);
  const [dir, setDir] = useState("rising");
  const [oldFrom, setOldFrom] = useState(2011);
  const [newFrom, setNewFrom] = useState(last - 4);
  const windows = useDebounced({ old: `${oldFrom}-${oldFrom + 4}`, new: `${newFrom}-${newFrom + 4}` }, 500);

  const terms = useApi(active.length ? "titles/terms" : null, { terms: active.join(","), from: fromD });
  const style = useApi("titles/style");
  const words = useApi("titles/words", { direction: dir, old: windows.old, new: windows.new, limit: 15 });

  const toggle = (t) => setActive((a) => (a.includes(t) ? a.filter((x) => x !== t) : a.length >= 8 ? a : [...a, t]));
  const addCustom = (e) => {
    e.preventDefault();
    const t = custom.trim().toLowerCase();
    if (t && !active.includes(t) && active.length < 8) setActive([...active, t]);
    setCustom("");
  };
  const colorOf = (t) => palette[active.indexOf(t) % palette.length];

  const termsOpt = useMemo(() => terms.data && lineOption(p, { labels: terms.data.years, fmtY: (v) => `${Number(v).toFixed(2)}%`,
    series: terms.data.series.map((s) => ({ name: termLabel(s.term), data: s.values, color: colorOf(s.term) })) }), [p, terms.data]);
  const styleOpt = useMemo(() => style.data && lineOption(p, { labels: style.data.map((r) => `${r.decade}s`), fmtY: fmt.pct, series: [
    { name: "Has a colon", data: style.data.map((r) => r.colon), color: p.s1 },
    { name: "“Name:” style", data: style.data.map((r) => r.name_colon), color: p.s2 },
    { name: "Is a question", data: style.data.map((r) => r.question), color: p.s3 },
    { name: "Starts with “toward(s)”", data: style.data.map((r) => r.towards), color: p.s6 },
  ] }), [p, style.data]);
  const wordsOpt = useMemo(() => {
    if (!words.data) return null;
    const f = (v) => (v >= 1 ? "×" + Number(v).toFixed(v >= 10 ? 0 : 1) : "÷" + (1 / v).toFixed(1));
    return hbarOption(p, { labels: words.data.map((r) => r.word), fmtX: f, labelWidth: 110,
      series: [{ name: dir === "rising" ? "Change" : "Change", data: words.data.map((r) => r.change_x), color: dir === "rising" ? p.s1 : p.bad }] });
  }, [p, words.data, dir]);

  return (
    <>
      <PageHead eyebrow="04 · What titles say" title="Titles are the only text dblp has">
        No abstracts, no citations: just the title. Type any word or phrase to trace it through journal and conference titles.
      </PageHead>
      <div className="grid">
        <Card span2 state={active.length ? terms : undefined} height={320} title="Research topics come in waves"
              sub={`Share of titles containing each term (up to 8), ${fromD}–${last}. Whole-word match; the report's terms keep their exact patterns (llm also matches “large language model”).`}
              controls={<>
                <ChipGroup>
                  {[...new Set([...PRESETS, ...active])].map((t) => (
                    <Chip key={t} label={termLabel(t)} on={active.includes(t)} color={active.includes(t) ? colorOf(t) : undefined} onClick={() => toggle(t)} />
                  ))}
                </ChipGroup>
                <form className="inlineform" onSubmit={addCustom}>
                  <input id="term-add" className="textinput" value={custom} maxLength={40} placeholder="Add a term, e.g. reinforcement learning"
                         aria-label="Add a term" onChange={(e) => setCustom(e.target.value)} />
                  <button type="submit" className="btn" disabled={!custom.trim() || active.length >= 8}>Add</button>
                </form>
                <FilterLabel htmlFor="term-from">From</FilterLabel>
                <RangeSlider id="term-from" label="First year" min={1970} max={last - 5} value={from} onChange={setFrom} />
              </>}>
          {() => (active.length ? <Chart option={termsOpt} height={320} label="Topic waves" /> : <EmptyNote>Pick at least one term.</EmptyNote>)}
        </Card>
        <Card state={style} title="Titles got longer and more branded" sub="Share of titles by decade.">
          {() => <Chart option={styleOpt} label="Title style" />}
        </Card>
        <Card state={words} height={360} title="Rising and falling title words"
              sub="Change in the share of titles using the word between two five-year windows (+5 smoothing, 1,500+ titles)."
              controls={<>
                <Seg label="Direction" value={dir} onChange={setDir} options={[{ v: "rising", l: "Rising" }, { v: "falling", l: "Falling" }]} />
                <FilterLabel htmlFor="w-old">Compare</FilterLabel>
                <NumberInput id="w-old" label="First window start" value={oldFrom} min={1970} max={newFrom - 5} onChange={setOldFrom} />
                <span className="muted">–{oldFrom + 4} with</span>
                <NumberInput id="w-new" label="Second window start" value={newFrom} min={oldFrom + 5} max={last - 4} onChange={setNewFrom} />
                <span className="muted">–{newFrom + 4}</span>
              </>}>
          {() => (words.data.length ? <Chart option={wordsOpt} height={360} label="Rising and falling words" />
            : <EmptyNote>No word crosses the 1,500-title threshold for these windows.</EmptyNote>)}
        </Card>
      </div>
    </>
  );
}

// ============================================================== Venues ====
const PERIOD_PRESETS = {
  report: { l: "2001–05 · 2011–15 · 2021–25", v: "2001-2005,2011-2015,2021-2025" },
  decades: { l: "1990s · 2000s · 2010s · 2020s", v: "1990-1999,2000-2009,2010-2019,2020-2025" },
};
const TREEMAP_PERIODS = ["2001-2005", "2011-2015", "2021-2025"];

export function PageVenues({ go }) {
  const p = usePalette();
  const [kind, setKind] = useState("conference");
  const [topN, setTopN] = useState(10);
  const [preset, setPreset] = useState("report");
  const [minPapers, setMinPapers] = useState(2000);
  const [maxDoi, setMaxDoi] = useState(1);
  const minP = useDebounced(minPapers, 600);
  const [treePeriod, setTreePeriod] = useState(TREEMAP_PERIODS[2]);
  const [scatterMin, setScatterMin] = useState(1000);

  const life = useApi("venues/lifespans");
  const conc = useApi("venues/concentration", { top: topN });
  const pubs = useApi("venues/publishers", { periods: PERIOD_PRESETS[preset].v });
  const gaps = useApi("venues/doi-gaps", { min_papers: minP, max_doi_pct: maxDoi, limit: 15 });
  const tree = useApi("venues/treemap", { period: treePeriod });
  const scat = useApi("venues/scatter", { min_papers: scatterMin });

  const lifeRows = life.data?.filter((r) => r.kind === kind);
  const lifeColor = kind === "conference" ? p.s1 : p.s2;
  const activeOpt = useMemo(() => lifeRows && lineOption(p, { labels: lifeRows.map((r) => `${r.started}s`), fmtY: fmt.pct,
    series: [{ name: "Still active", data: lifeRows.map((r) => r.pct_still_active), color: lifeColor }] }), [p, life.data, kind]);
  const yearsOpt = useMemo(() => lifeRows && lineOption(p, { labels: lifeRows.map((r) => `${r.started}s`), fmtY: fmt.fixed1,
    series: [{ name: "Median active years", data: lifeRows.map((r) => r.median_active_years), color: lifeColor }] }), [p, life.data, kind]);
  const concOpt = useMemo(() => conc.data && lineOption(p, { labels: conc.data.map((r) => r.year), fmtY: fmt.pct, series: [
    { name: `Top ${topN} conferences`, data: conc.data.map((r) => r.conf_top_pct), color: p.s1 },
    { name: `Top ${topN} journals`, data: conc.data.map((r) => r.journal_top_pct), color: p.s2 },
  ] }), [p, conc.data]);
  const pubOpt = useMemo(() => {
    if (!pubs.data) return null;
    const colors = [p.s4, p.s6, p.s1, p.s3];
    return barOption(p, { labels: pubs.data.rows.map((r) => r.publisher), fmtY: fmt.pct,
      series: pubs.data.periods.map((per, i) => ({ name: per, data: pubs.data.rows.map((r) => r[`pct_${i}`]), color: colors[i] })) });
  }, [p, pubs.data]);
  const gapOpt = useMemo(() => gaps.data && hbarOption(p, { labels: gaps.data.map((r) => r.usual_name), fmtX: fmt.comma, labelWidth: 190,
    series: [{ name: "Papers", data: gaps.data.map((r) => r.papers), color: p.bad }] }), [p, gaps.data]);
  const treeOpt = useMemo(() => {
    if (!tree.data) return null;
    const total = tree.data.children.reduce((s, c) => s + c.value, 0);
    return treemapOption(p, { children: tree.data.children, tooltip: (q) => {
      const pub = q.treePathInfo?.[1]?.name;
      const share = q.treePathInfo?.length > 2 ? `${(100 * q.value / (q.treePathInfo[1].value || 1)).toFixed(1)}% of ${pub}` : `${(100 * q.value / total).toFixed(1)}% of all`;
      return `<b>${q.name}</b><br>${fmt.comma(q.value)} papers · ${share}`; } });
  }, [p, tree.data]);
  const scatOpt = useMemo(() => scat.data && scatterOption(p, {
    xName: "papers in the series (log)", yName: "open access", fmtX: fmt.comma, fmtY: fmt.pct,
    series: [["journal", "Journals", p.s2], ["conference", "Conferences", p.s1]].map(([k, name, color]) => ({
      name, color, data: scat.data.filter((r) => r.kind === k).map((r) => ({ x: r.papers, y: r.pct_oa, name: r.name, sid: r.sid, doi: r.pct_doi })) })),
    tooltip: (q) => `<b>${q.data.name}</b><br>${fmt.comma(q.data.x)} papers · ${q.data.y}% open access · ${q.data.doi}% with a DOI`,
  }), [p, scat.data]);

  return (
    <>
      <PageHead eyebrow="05 · Venues and publishers" title="Conferences come and go; journals last">
        Venues are identified by dblp’s stable series key (the part of a record key like <code>conf/cvpr</code>), not the fragmented venue-name string.
      </PageHead>
      <div className="grid">
        <Card state={life} title="Series still active" sub="Share with papers in the last two complete years, by the decade the series started."
              controls={<Seg label="Kind" value={kind} onChange={setKind} options={[{ v: "conference", l: "Conferences" }, { v: "journal", l: "Journals" }]} />}>
          {() => <Chart option={activeOpt} label="Series still active" />}
        </Card>
        <Card state={life} title="Series lifespan" sub="Median number of years with papers, by starting decade."
              controls={<span className="hint">{cap(kind)}s · switch on the left</span>}>
          {() => <Chart option={yearsOpt} label="Series lifespan" />}
        </Card>
        <Card span2 state={conc} height={270} title="Publishing spread out, then concentrated again"
              sub="Share of each year’s papers that appear in that year’s largest series."
              controls={<><FilterLabel>Largest</FilterLabel><Seg label="Number of series" value={topN} onChange={setTopN} options={[5, 10, 20, 50].map((n) => ({ v: n, l: String(n) }))} /></>}>
          {() => <Chart option={concOpt} height={270} label="Concentration" />}
        </Card>
        <Card span2 state={pubs} height={280} title="Who publishes it" sub="Share of journal and conference papers by publisher, identified from the DOI prefix."
              controls={<><FilterLabel>Periods</FilterLabel><Seg label="Periods" value={preset} onChange={setPreset} options={Object.entries(PERIOD_PRESETS).map(([k, v]) => ({ v: k, l: v.l }))} /></>}>
          {() => <Chart option={pubOpt} height={280} label="Publishers" />}
        </Card>
        <Card span2 state={tree} height={420} title="Inside each publisher" sub="Area is papers in the period: each publisher, and the largest series within it. Click a series to explore it."
              controls={<><FilterLabel>Period</FilterLabel><Seg label="Period" value={treePeriod} onChange={setTreePeriod} options={TREEMAP_PERIODS.map((v) => ({ v, l: v.replace("-", "–") }))} /></>}>
          {() => <Chart option={treeOpt} height={420} label="Publishers and their series" onClick={(e) => { if (e.data?.sid) go("venues", { sid: e.data.sid }); }} />}
        </Card>
        <Card span2 state={scat} height={380} title="The biggest venues are the open ones"
              sub="Every active series above the size threshold: how large it is against how much of it is flagged open access. The mega-journals sit top right. Click a point to explore the venue."
              controls={<><FilterLabel htmlFor="scat-min">Min. papers</FilterLabel>
                <NumberInput id="scat-min" label="Minimum papers" value={scatterMin} min={100} max={100000} step={100} onChange={setScatterMin} width={90} /></>}>
          {() => (scat.data.length
            ? <Chart option={scatOpt} height={380} label="Series size against open-access share" onClick={(e) => { if (e.data?.sid) go("venues", { sid: e.data.sid }); }} />
            : <EmptyNote>No series above this size.</EmptyNote>)}
        </Card>
        <Card span2 state={gaps} height={380} title="Where the DOI bridge is out"
              sub="Largest series whose papers almost never carry a DOI, so they can’t be enriched through OpenAlex. Click a bar to explore the venue."
              controls={<>
                <FilterLabel htmlFor="gap-min">Min. papers</FilterLabel>
                <NumberInput id="gap-min" label="Minimum papers" value={minPapers} min={100} max={100000} step={100} onChange={setMinPapers} width={90} />
                <FilterLabel>DOI coverage below</FilterLabel>
                <Seg label="DOI coverage below" value={maxDoi} onChange={setMaxDoi} options={[1, 10, 50].map((n) => ({ v: n, l: `${n}%` }))} />
              </>}>
          {() => (gaps.data.length
            ? <Chart option={gapOpt} height={380} label="DOI gaps" onClick={(e) => { const r = gaps.data.find((x) => x.usual_name === e.name); if (r) go("venues", { sid: r.sid }); }} />
            : <EmptyNote>No series matches these thresholds.</EmptyNote>)}
        </Card>
      </div>
    </>
  );
}

// ============================================================= Quality ====
export function PageQuality() {
  const p = usePalette();
  const cov = useApi("quality/coverage");
  const pages = useApi("quality/page-formats");
  const theses = useApi("quality/theses");
  const rd = useApi("quality/research-data");
  const [off, setOff] = useState({});
  const [typesOff, setTypesOff] = useState({ mastersthesis: true });

  const pageOpt = useMemo(() => pages.data && hbarOption(p, { labels: pages.data.map((r) => r.page_format), fmtX: fmt.comma, labelWidth: 170,
    series: [{ name: "Papers", data: pages.data.map((r) => r.papers), color: p.s1 }] }), [p, pages.data]);
  const thesisOpt = useMemo(() => theses.data && hbarOption(p, { labels: theses.data.countries.map((r) => r.country), fmtX: fmt.pct1, labelWidth: 100,
    series: [{ name: "Share of theses", data: theses.data.countries.map((r) => r.pct), color: p.s3 }] }), [p, theses.data]);
  const thesisYearOpt = useMemo(() => theses.data && lineOption(p, { labels: theses.data.by_year.map((r) => r.year), fmtY: fmt.comma,
    series: [{ name: "PhD theses", data: theses.data.by_year.map((r) => r.theses), color: p.s3 }] }), [p, theses.data]);
  const rdOpt = useMemo(() => rd.data && barOption(p, { labels: rd.data.by_year.map((r) => r.year), stacked: true, fmtY: fmt.comma, series: [
    { name: "Plain records", data: rd.data.by_year.map((r) => r.plain), color: p.s6 },
    { name: "Versions", data: rd.data.by_year.map((r) => r.versions), color: p.s5 },
    { name: "Concepts", data: rd.data.by_year.map((r) => r.concepts), color: p.s4 },
  ] }), [p, rd.data]);

  return (
    <>
      <PageHead eyebrow="06 · Records and data quality" title="Every record type has its own schema">
        A conference paper never has a journal; a thesis always has a school. “Missingness” only means something computed per type.
      </PageHead>
      <div className="grid">
        <Card span2 state={cov} height={420} title="Field coverage by record type"
              sub="Share of all records of each type that carry the field, straight from the full dump. Toggle rows and columns."
              controls={cov.data && <>
                <FilterLabel>Types</FilterLabel>
                <ChipGroup>{cov.data.types.map((t) => <Chip key={t} label={t} on={!typesOff[t]} onClick={() => setTypesOff({ ...typesOff, [t]: !typesOff[t] })} />)}</ChipGroup>
                <FilterLabel>Fields</FilterLabel>
                <ChipGroup>{cov.data.fields.map((f) => <Chip key={f} label={f} on={!off[f]} onClick={() => setOff({ ...off, [f]: !off[f] })} />)}</ChipGroup>
              </>}>
          {(d) => {
            const cols = d.types.map((t, i) => ({ t, i })).filter((x) => !typesOff[x.t]);
            const rows = d.fields.map((f, i) => ({ f, i })).filter((x) => !off[x.f]);
            return (
              <>
                <Heatmap colLabels={cols.map((c) => `${c.t} (${fmt.comma(d.records[c.i])})`)} rowLabels={rows.map((r) => r.f)}
                         matrix={rows.map((r) => cols.map((c) => d.matrix[r.i][c.i]))} />
              </>
            );
          }}
        </Card>
        <Card state={pages} height={260} title="The pages field is free text" sub="Format actually used across journal and conference papers.">
          {() => <Chart option={pageOpt} height={260} label="Page formats" />}
        </Card>
        <Card state={theses} height={320} title="PhD theses by country" sub="Last part of the school name. Follows which national libraries dblp harvests, not where research happens.">
          {() => <Chart option={thesisOpt} height={320} label="Theses by country" />}
        </Card>
        <Card state={theses} title="PhD theses per year" sub="Harvesting waves, not a research trend.">
          {() => <Chart option={thesisYearOpt} label="Theses per year" />}
        </Card>
        <Card state={rd} title="Research data records" sub={rd.data?.hosts?.length ? `The newest record type. Top link host: ${rd.data.hosts[0].link_host} (${rd.data.hosts[0].publisher}).` : "The newest record type."}>
          {() => <Chart option={rdOpt} label="Research data records" />}
        </Card>
      </div>
    </>
  );
}

// ========================================================== Enrichment ====
export function PageEnrichment() {
  const p = usePalette();
  const oa = useApi("enrichment/openalex");
  const d = oa.data?.available ? oa.data : null;
  const periods = d ? [...new Set(d.adds.map((a) => a.period))] : [];
  const [period, setPeriod] = useState(null);
  const per = period || periods[periods.length - 1];
  const kinds = ["journal", "conference", "preprint"];
  const covOpt = useMemo(() => {
    if (!d || !per) return null;
    const add = (k) => d.adds.find((a) => a.kind === k && a.period === per) || {};
    return hbarOption(p, { labels: ["Found in OpenAlex", "Has an abstract", "Has an institution"], fmtX: fmt.pct1, labelWidth: 130,
      series: kinds.map((k, i) => ({ name: cap(k), color: [p.s1, p.s2, p.s3][i],
        data: [d.match[k]?.found, add(k).pct_abstract, add(k).pct_institution] })) });
  }, [p, d, per]);
  const fieldOpt = useMemo(() => d && hbarOption(p, { labels: d.fields.map((r) => r.field), fmtX: fmt.pct1, labelWidth: 220,
    series: [{ name: "Share of papers", data: d.fields.map((r) => r.pct), color: p.s4 }] }), [p, d]);
  return (
    <>
      <PageHead eyebrow="07 · Checked against OpenAlex" title="How good is the DOI bridge?">
        700 random DOIs per kind, looked up in OpenAlex, the open catalogue with the abstracts, citations and institutions dblp
        lacks. It calls an external API, so this page shows the latest run of that job. Each figure is accurate to about ±4 points.
      </PageHead>
      {oa.data && !d ? <Callout>{oa.data.reason}</Callout> : null}
      <div className="grid">
        <Card state={oa} job={oa.data?.source} height={280} title="What linking through DOIs would add"
              sub="“Found” covers the whole sample; abstract and institution shares are for the selected publication period."
              controls={periods.length > 1 && <Seg label="Period" value={per} onChange={setPeriod} options={periods.map((x) => ({ v: x, l: x }))} />}>
          {() => d && <Chart option={covOpt} height={280} label="OpenAlex coverage" />}
        </Card>
        <Card state={oa} job={oa.data?.source} height={320} title="Only about half is primarily computer science" sub="OpenAlex’s primary field for dblp papers.">
          {() => d && <Chart option={fieldOpt} height={320} label="OpenAlex fields" />}
        </Card>
        <Card span2 state={oa} job={oa.data?.source} title="Agreement between dblp and OpenAlex" sub="Share of matched works where the two catalogues agree.">
          {() => d && (
            <SortableTable defaultSort={{ key: "kind", dir: 1 }} rows={kinds.map((k) => ({ kind: k, ...d.match[k] }))} columns={[
              { key: "kind", label: "Kind", render: (r) => cap(r.kind) },
              { key: "found", label: "Found", num: true, render: (r) => pct(r.found) },
              { key: "year_exact", label: "Same year", num: true, render: (r) => pct(r.year_exact) },
              { key: "year_within_1", label: "Year ±1", num: true, render: (r) => pct(r.year_within_1) },
              { key: "title_close", label: "Title close", num: true, render: (r) => pct(r.title_close) },
              { key: "author_count_equal", label: "Same author count", num: true, render: (r) => pct(r.author_count_equal) },
            ]} />
          )}
        </Card>
      </div>
      <Callout><b>dblp’s scope is “publishes in CS venues,”</b> a much wider net than “is CS research.”</Callout>
    </>
  );
}

// =============================================================== Tails ====
const PANELS = [
  { v: "papers_per_author", l: "Papers per author" },
  { v: "coauthors_per_author", l: "Co-authors per author" },
  { v: "papers_per_series", l: "Papers per venue series" },
];

export function PageTails() {
  const p = usePalette();
  const [panel, setPanel] = useState(PANELS[0].v);
  const t = useApi("tails", { panel });
  const color = { papers_per_author: p.s1, coauthors_per_author: p.s2, papers_per_series: p.s3 }[panel];
  const d = t.data;
  const opt = useMemo(() => d?.points?.length && loglogOption(p, { name: d.label, color, data: d.points, xLabel: d.label }), [p, d]);
  const s = d?.stats;
  const fit = d?.fit;
  const joint = useApi("tails/joint", { bins_per_decade: 6 });
  const jointOpt = useMemo(() => joint.data && densityOption(p, {
    grid: joint.data.grid, nx: joint.data.nx, ny: joint.data.ny, binsPerDecade: joint.data.bins_per_decade,
    xName: "papers", yName: "distinct co-authors", levels: [10, 100, 1000, 10000, 100000], total: joint.data.authors_plotted,
  }), [p, joint.data]);
  return (
    <>
      <PageHead eyebrow="08 · The long tail, formally" title="Heavy-tailed, but not power laws">
        Most authors have very little and a few have a lot. But a lognormal fits these distributions better than a pure power law.
      </PageHead>
      <Filters page><FilterLabel>Distribution</FilterLabel><Seg label="Distribution" options={PANELS} value={panel} onChange={setPanel} /></Filters>
      <KpiStrip loading={!s} items={s && [
        { n: fmt.comma(s.n), l: "entities (disambiguation bins excluded)" },
        { n: `${s.median}`, l: `median · mean ${s.mean}` },
        { n: fmt.comma(s.max), l: "maximum" },
        { n: `${s.gini}`, l: "Gini coefficient" },
        { n: `${s.top1_share}%`, l: "of the total held by the top 1%" },
        { n: s.alpha_live ? `α ≈ ${s.alpha_live}` : "—", l: fit?.xmin ? `power-law exponent above x = ${fmt.comma(fit.xmin)} (all data)` : "power-law fit unavailable" },
      ]} />
      <div className="grid">
        <Card span2 state={t} height={380} title={`Share with at least x: ${(d?.label || "").toLowerCase()}`}
              sub="Log–log. A straight line would be a power law; the curve bends down instead.">
          {() => (d.points.length ? <Chart option={opt} height={380} label="Complementary CDF" /> : <EmptyNote>No data for this distribution.</EmptyNote>)}
        </Card>
        <Card span2 state={t} job={d?.fit_source} title="Is the tail a power law?"
              sub="Likelihood-ratio tests from the statistics job (Clauset–Shalizi–Newman). R < 0 favours the alternative.">
          {() => (fit?.comparisons ? (
            <SortableTable defaultSort={{ key: "alt", dir: 1 }} rows={Object.entries(fit.comparisons).map(([alt, v]) => ({ alt, v }))} columns={[
              { key: "alt", label: "Power law vs.", render: (r) => r.alt.replace(/_/g, " ") },
              { key: "v", label: "R (p-value)", render: (r) => <span className="num">{r.v}</span> },
            ]} />
          ) : <EmptyNote>The statistics job has not produced a fit for this distribution.</EmptyNote>)}
        </Card>
        <Card span2 state={joint} height={420} title="Papers against co-authors, every author at once"
              sub={joint.data ? `Where ${fmt.comma(joint.data.authors_plotted)} authors sit on both scales, as a density on log–log axes (colour is the count in each cell, on a log ramp; lines are iso-counts of 10, 100, 1,000, 10,000 and 100,000). ${pct(Math.round(100 * joint.data.left_out_share))} of authors have no co-author and cannot sit on a log axis. Hover a cell for its count.` : "The joint distribution behind the two author panels above."}>
          {() => <Chart option={jointOpt} height={420} label="Density of papers against co-authors" />}
        </Card>
      </div>
      <Callout>
        The job fitted a 300,000-item sample{fit?.alpha ? ` (α = ${fit.alpha} ± ${fit.alpha_se} above x = ${fit.xmin})` : ""}; the α above is
        recomputed on all data at the job’s x<sub>min</sub>. Either way: heavy-tailed, not scale-free.
      </Callout>
    </>
  );
}
