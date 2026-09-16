import { useMemo, useState } from "react";
import D from "./data.json";
import { Chart, usePalette, lineOption, barOption, hbarOption, loglogOption, growthOption, fmt } from "./charts.jsx";
import { KpiStrip, PageHead, Card, Filters, FilterLabel, Chip, ChipGroup, Seg, RangeSlider, SortableTable, Heatmap, Callout } from "./components.jsx";

const cap = (s) => s[0].toUpperCase() + s.slice(1);

// ============================================================ Overview ====
export function PageOverview() {
  const p = usePalette();
  const g = D.growth_by_kind;
  const growth = useMemo(() => lineOption(p, { labels: g.map((r) => r.year), log: true, fmtY: fmt.comma, series: [
    { name: "Conference", data: g.map((r) => r.conference), color: p.s1 },
    { name: "Journal", data: g.map((r) => r.journal), color: p.s2 },
    { name: "Preprint", data: g.map((r) => r.preprint), color: p.s3 },
  ] }), [p]);
  const hom = useMemo(() => hbarOption(p, { labels: D.top_homonyms.slice(0, 10).map((r) => r.base_name), fmtX: fmt.comma, labelWidth: 90,
    series: [{ name: "People", data: D.top_homonyms.slice(0, 10).map((r) => r.distinct_people), color: p.s1 }] }), [p]);
  const net = useMemo(() => lineOption(p, { labels: D.network_growth.map((r) => r.up_to), fmtY: fmt.pct,
    series: [{ name: "Share in giant component", data: D.network_growth.map((r) => r.giant_pct), color: p.s3 }] }), [p]);
  return (
    <>
      <PageHead eyebrow="dblp · 2026-09-01 snapshot · 12,928,181 records" title="Reading dblp">
        dblp looks like a list of CS papers. It's really two datasets braided together: 8.7M publications, and a 4.2M-person registry deciding who wrote them. Every finding is explorable below — filters are live.
      </PageHead>
      <KpiStrip items={D.kpis} />
      <div className="grid">
        <Card span2 title="Papers per year, by kind" sub="1970–2025, log scale. Conferences overtook journals in 1990; journals retook the lead in 2020 as mega-journals grew.">
          <Chart option={growth} height={300} />
        </Card>
        <Card title="One name, hundreds of people" sub="Distinct author pages sharing a base name, top 10.">
          <Chart option={hom} height={300} />
        </Card>
        <Card title="The co-authorship network" sub="94.5% of connected authors sit in one giant, small-world component.">
          <Chart option={net} height={300} />
        </Card>
      </div>
    </>
  );
}

// ========================================================== Publishing ====
export function PagePublishing() {
  const p = usePalette();
  const [range, setRange] = useState([1970, 2025]);
  const [kinds, setKinds] = useState({ conference: true, journal: true, preprint: true });
  const kindColor = { conference: p.s1, journal: p.s2, preprint: p.s3 };
  const g = useMemo(() => D.growth_by_kind.filter((r) => r.year >= range[0] && r.year <= range[1]), [range]);
  const t = useMemo(() => D.team_size.filter((r) => r.year >= range[0] && r.year <= range[1]), [range]);
  const m = D.metadata_trends;

  const growth = useMemo(() => lineOption(p, { labels: g.map((r) => r.year), log: true, fmtY: fmt.comma,
    series: Object.keys(kinds).filter((k) => kinds[k]).map((k) => ({ name: cap(k), data: g.map((r) => r[k]), color: kindColor[k] })) }), [p, g, kinds]);
  const team = useMemo(() => lineOption(p, { labels: t.map((r) => r.year), fmtY: fmt.fixed1,
    series: [{ name: "Mean authors", data: t.map((r) => r.mean_authors), color: p.s1 }] }), [p, t]);
  const solo = useMemo(() => lineOption(p, { labels: t.map((r) => r.year), fmtY: fmt.frac,
    series: [{ name: "Single-author share", data: t.map((r) => r.single_author_share), color: p.s2 }] }), [p, t]);
  const meta = useMemo(() => lineOption(p, { labels: m.map((r) => r.year), fmtY: fmt.frac, series: [
    { name: "Has an ORCID", data: m.map((r) => r.with_orcid), color: p.s1 },
    { name: "Unidentified author", data: m.map((r) => r.with_unidentified_author), color: p.bad },
    { name: "Preprint/published twin", data: m.map((r) => r.with_title_twin), color: p.s4 },
    { name: "Open access", data: m.map((r) => r.open_access), color: p.s3 },
  ] }), [p]);
  const rates = useMemo(() => growthOption(p, D.growth_rates), [p]);

  return (
    <>
      <PageHead eyebrow="01 · Fifty years of publishing" title="Growth, teams, and where the papers go">
        CS publishing grows ~9.5% a year, doubling every 7.6 years — but conferences, journals and preprints tell very different stories.
      </PageHead>
      <Filters page>
        <FilterLabel>Years</FilterLabel>
        <RangeSlider id="yr-from" min={1970} max={range[1]} value={range[0]} onChange={(v) => setRange([v, range[1]])} />
        <span style={{ color: "var(--muted)", fontSize: 12 }}>to</span>
        <RangeSlider id="yr-to" min={range[0]} max={2025} value={range[1]} onChange={(v) => setRange([range[0], v])} />
        <FilterLabel>Kind</FilterLabel>
        <ChipGroup>
          {Object.keys(kinds).map((k) => <Chip key={k} label={cap(k)} on={kinds[k]} color={kindColor[k]} onClick={() => setKinds({ ...kinds, [k]: !kinds[k] })} />)}
        </ChipGroup>
      </Filters>
      <div className="grid">
        <Card span2 title="Papers per year, by kind" sub="Log scale — a straight line is steady percentage growth."><Chart option={growth} height={300} /></Card>
        <Card title="Mean authors per paper" sub="1.4 in 1970 → 4.4 in 2025. Preprints excluded."><Chart option={team} /></Card>
        <Card title="Share of single-author papers" sub="69% in 1970 → 5% today."><Chart option={solo} /></Card>
        <Card span2 title="Four metadata trends, 2000–2025" sub="ORCID and open access are rising; so are unidentified authors and duplicate preprints."><Chart option={meta} height={290} /></Card>
        <Card span2 title="Annual growth rate" sub="Log-linear fit with 95% intervals (optimistic — years aren't independent). Hover for the interval and doubling time."><Chart option={rates} height={270} /></Card>
      </div>
    </>
  );
}

// ============================================================ Identity ====
export function PageIdentity() {
  const p = usePalette();
  const [homN, setHomN] = useState(10);
  const hom = useMemo(() => hbarOption(p, { labels: D.top_homonyms.slice(0, homN).map((r) => r.base_name), fmtX: fmt.comma, labelWidth: 90,
    series: [{ name: "People", data: D.top_homonyms.slice(0, homN).map((r) => r.distinct_people), color: p.s1 }] }), [p, homN]);
  const aff = useMemo(() => hbarOption(p, { labels: ["Has affiliation", "Links ORCID", "Links Wikidata"], fmtX: fmt.frac, labelWidth: 110, series: [
    { name: "Regular pages", color: p.s1, data: ["affiliation", "orcid", "wikidata"].map((k) => D.affiliation_effect.regular[k]) },
    { name: "Numbered namesakes", color: p.s4, data: ["affiliation", "orcid", "wikidata"].map((k) => D.affiliation_effect.numbered[k]) },
  ] }), [p]);
  const newc = useMemo(() => lineOption(p, { labels: D.newcomers.map((r) => r.year), fmtY: fmt.comma,
    series: [{ name: "First-time authors", data: D.newcomers.map((r) => r.new_authors), color: p.s3 }] }), [p]);
  const cohort = useMemo(() => lineOption(p, { labels: D.cohort_survival.map((r) => r.cohort), fmtY: fmt.pct, series: [
    { name: "Still publishing 5 y later", data: D.cohort_survival.map((r) => r.pct_publishing_5y_later), color: p.s1 },
    { name: "10 y later", data: D.cohort_survival.map((r) => r.pct_10y_later), color: p.s2 },
    { name: "20 y later", data: D.cohort_survival.map((r) => r.pct_20y_later), color: p.s6 },
  ] }), [p]);
  const pos = useMemo(() => hbarOption(p, { labels: D.author_position.map((r) => r.author_total_papers + " papers"), stacked: true, fmtX: fmt.pct, labelWidth: 110, series: [
    { name: "First author", color: p.s1, data: D.author_position.map((r) => r.pct_first) },
    { name: "Middle", color: p.muted, data: D.author_position.map((r) => r.pct_middle) },
    { name: "Last author", color: p.s2, data: D.author_position.map((r) => r.pct_last) },
  ] }), [p]);
  const alpha = useMemo(() => {
    const rows = [...D.alphabetical_order].sort((a, b) => b.pct_alphabetical - a.pct_alphabetical);
    return hbarOption(p, { labels: rows.map((r) => r.usual_name), fmtX: fmt.pct, labelWidth: 210, series: [
      { name: "Actually alphabetical", color: p.s1, data: rows.map((r) => r.pct_alphabetical) },
      { name: "Expected by chance", color: p.muted, data: rows.map((r) => r.pct_by_chance) },
    ] });
  }, [p]);

  return (
    <>
      <PageHead eyebrow="02 · Author identity" title="One name, hundreds of people">
        dblp stores no author ID on papers — only a name string, resolved through a 4.19M-page registry that fails in three ways: name variants, numbered namesakes, and 33,659 unassigned disambiguation bins.
      </PageHead>
      <div className="grid">
        <Card title="Most-shared names" sub="Distinct author pages per base name.">
          <Filters><FilterLabel>Show top</FilterLabel><RangeSlider id="hom-n" min={5} max={15} value={homN} onChange={setHomN} /></Filters>
          <Chart option={hom} height={homN > 10 ? 340 : 280} />
        </Card>
        <Card title="Affiliations exist to tell people apart" sub="100% of numbered namesakes carry an affiliation; 1.7% of everyone else does.">
          <Chart option={aff} height={280} />
        </Card>
        <Card title="New authors per year" sub="230,856 people published their first dblp-indexed paper in 2025."><Chart option={newc} /></Card>
        <Card title="Most authors do not stay" sub="Share of each starting cohort still publishing k years later (later cohorts are cut off by the data)."><Chart option={cohort} /></Card>
        <Card span2 title="Who signs where" sub="Position on papers with 3+ authors, by the author's career total. Prolific authors sign last."><Chart option={pos} height={240} /></Card>
        <Card span2 title="Alphabetical author order" sub="Theory and math venues list authors alphabetically 90–95% of the time; medical imaging sits at chance."><Chart option={alpha} height={420} /></Card>
      </div>
    </>
  );
}

// ============================================================= Network ====
export function PageNetwork() {
  const p = usePalette();
  const growth = useMemo(() => lineOption(p, { labels: D.network_growth.map((r) => r.up_to), fmtY: fmt.pct,
    series: [{ name: "Share in giant component", data: D.network_growth.map((r) => r.giant_pct), color: p.s3 }] }), [p]);
  const degree = useMemo(() => lineOption(p, { labels: D.network_growth.map((r) => r.up_to), fmtY: fmt.fixed1,
    series: [{ name: "Mean co-authors", data: D.network_growth.map((r) => r.mean_degree), color: p.s1 }] }), [p]);
  const dist = useMemo(() => barOption(p, { labels: D.distances.map((r) => r.hops + (r.hops === 1 ? " hop" : " hops")), fmtY: fmt.pct,
    series: [{ name: "Share of pairs", data: D.distances.map((r) => r.pct_of_pairs), color: p.s1 }] }), [p]);
  return (
    <>
      <PageHead eyebrow="03 · The co-authorship network" title="A small world, built from 26.8M author pairs">
        Two authors are linked if they share a paper with 2–50 authors. Disambiguation bins are removed first — that alone strips 17% of all edges.
      </PageHead>
      <KpiStrip items={[
        { n: "94.5%", l: "of connected authors sit in one giant component" },
        { n: "11.6", l: "mean distinct co-authors (median 5)" },
        { n: "5.63", l: "average co-authorship steps between two people" },
        { n: "0.661", l: "average local clustering — your co-authors know each other" },
        { n: "k = 49", l: "densest core: 154 authors, each with 49+ co-authors inside it" },
        { n: "2,202", l: "Leiden communities, modularity 0.734" },
      ]} />
      <div className="grid">
        <Card title="How the network grew" sub="Share of connected authors in the largest component, 1970–2025."><Chart option={growth} /></Card>
        <Card title="Mean distinct co-authors" sub="From 2.2 in 1970 to 11.4 in 2025."><Chart option={degree} /></Card>
        <Card span2 title="Six degrees of separation" sub="Distances from 20 random authors to everyone else in the giant component. Mean 5.63, median 6, 90% within 7."><Chart option={dist} height={250} /></Card>
        <Card span2 title="The largest communities" sub="Leiden detection. Click a column to sort; each community is labelled by its members' top venues.">
          <SortableTable defaultSort={{ key: "authors", dir: -1 }} rows={D.communities} columns={[
            { key: "authors", label: "Authors", num: true, render: (r) => fmt.comma(r.authors) },
            { key: "venues", label: "Top venues", render: (r) => r.venues.map((v, i) => <span key={i} className="venuepill">{v.name} ({fmt.comma(v.papers)})</span>) },
          ]} />
        </Card>
      </div>
    </>
  );
}

// ============================================================== Titles ====
const TERMS = ["llm", "deep", "neural", "transformer", "cloud", "iot", "blockchain", "quantum"];
const TERM_LABEL = { llm: "LLM", deep: "Deep", neural: "Neural", transformer: "Transformer", cloud: "Cloud", iot: "IoT", blockchain: "Blockchain", quantum: "Quantum" };

export function PageTitles() {
  const p = usePalette();
  const termColor = { llm: p.s1, deep: p.s2, neural: p.s3, transformer: p.s4, cloud: p.s5, iot: p.s6, blockchain: p.bad, quantum: p.good };
  const [on, setOn] = useState({ llm: true, deep: true, neural: true, transformer: true, cloud: false, iot: false, blockchain: false, quantum: false });
  const [dir, setDir] = useState("rising");
  const waves = useMemo(() => lineOption(p, { labels: D.topic_waves.map((r) => r.year), fmtY: fmt.pct1,
    series: TERMS.filter((t) => on[t]).map((t) => ({ name: TERM_LABEL[t], data: D.topic_waves.map((r) => r[t]), color: termColor[t] })) }), [p, on]);
  const style = useMemo(() => lineOption(p, { labels: D.title_style.map((r) => r.decade + "s"), fmtY: fmt.pct, series: [
    { name: "Has a colon", data: D.title_style.map((r) => r.colon), color: p.s1 },
    { name: "“Name:” style", data: D.title_style.map((r) => r.name_colon), color: p.s2 },
    { name: "Is a question", data: D.title_style.map((r) => r.question), color: p.s3 },
  ] }), [p]);
  const words = useMemo(() => {
    const rows = D.rising_falling_words.filter((r) => (dir === "rising" ? r.change_x >= 1 : r.change_x < 1))
      .sort((a, b) => (dir === "rising" ? b.change_x - a.change_x : a.change_x - b.change_x)).slice(0, 12);
    const f = (v) => (v >= 1 ? "×" + Number(v).toFixed(v >= 10 ? 0 : 1) : "÷" + (1 / v).toFixed(1));
    return hbarOption(p, { labels: rows.map((r) => r.word), fmtX: f, labelWidth: 100,
      series: [{ name: dir === "rising" ? "More common" : "Less common", data: rows.map((r) => r.change_x), color: dir === "rising" ? p.s1 : p.bad }] });
  }, [p, dir]);
  return (
    <>
      <PageHead eyebrow="04 · What titles say" title="Titles are the only text dblp has">
        No abstracts, no citations — just the title. Even so, eleven words per paper trace every research wave since 2000.
      </PageHead>
      <div className="grid">
        <Card span2 title="Research topics come in waves" sub="Share of journal and conference paper titles containing each term, 2000–2025.">
          <Filters><FilterLabel>Terms</FilterLabel><ChipGroup>
            {TERMS.map((t) => <Chip key={t} label={TERM_LABEL[t]} on={on[t]} color={termColor[t]} onClick={() => setOn({ ...on, [t]: !on[t] })} />)}
          </ChipGroup></Filters>
          <Chart option={waves} height={300} />
        </Card>
        <Card title="Titles got longer and more branded" sub="Share of titles by decade. Mean length grew from 7.9 to 11.2 words."><Chart option={style} /></Card>
        <Card title="Rising and falling title words" sub="Change in share of titles, 2011–15 vs. 2021–25.">
          <Filters><Seg options={[{ v: "rising", l: "Rising" }, { v: "falling", l: "Falling" }]} value={dir} onChange={setDir} /></Filters>
          <Chart option={words} height={330} />
        </Card>
      </div>
    </>
  );
}

// ============================================================== Venues ====
export function PageVenues() {
  const p = usePalette();
  const [kind, setKind] = useState("conference");
  const life = useMemo(() => {
    const rows = D.series_lifespans.filter((r) => r.kind === kind);
    return lineOption(p, { labels: rows.map((r) => r.started + "s"), fmtY: fmt.pct,
      series: [{ name: "Still active in 2024–25", data: rows.map((r) => r.pct_still_active), color: kind === "conference" ? p.s1 : p.s2 }] });
  }, [p, kind]);
  const lifeYears = useMemo(() => {
    const rows = D.series_lifespans.filter((r) => r.kind === kind);
    return lineOption(p, { labels: rows.map((r) => r.started + "s"), fmtY: fmt.fixed1,
      series: [{ name: "Median active years", data: rows.map((r) => r.median_active_years), color: kind === "conference" ? p.s1 : p.s2 }] });
  }, [p, kind]);
  const conc = useMemo(() => lineOption(p, { labels: D.concentration.map((r) => r.year), fmtY: fmt.pct, series: [
    { name: "Top 10 conferences", data: D.concentration.map((r) => r.conf_top10_pct), color: p.s1 },
    { name: "Top 10 journals", data: D.concentration.map((r) => r.journal_top10_pct), color: p.s2 },
  ] }), [p]);
  const pub = useMemo(() => barOption(p, { labels: D.publishers.map((r) => r.publisher), fmtY: fmt.pct, series: [
    { name: "2001–05", data: D.publishers.map((r) => r.pct_2001_05), color: p.s4 },
    { name: "2011–15", data: D.publishers.map((r) => r.pct_2011_15), color: p.s6 },
    { name: "2021–25", data: D.publishers.map((r) => r.pct_2021_25), color: p.s1 },
  ] }), [p]);
  const doi = useMemo(() => hbarOption(p, { labels: D.doi_gaps.map((r) => r.usual_name), fmtX: fmt.comma, labelWidth: 170,
    series: [{ name: "Papers", data: D.doi_gaps.map((r) => r.papers), color: p.bad }] }), [p]);
  return (
    <>
      <PageHead eyebrow="05 · Venues and publishers" title="Conferences come and go; journals last">
        6,589 conference series and 1,901 journals, identified by dblp's own stable series key — not the fragmented venue-name string.
      </PageHead>
      <Filters page><FilterLabel>Lifespans for</FilterLabel><Seg options={[{ v: "conference", l: "Conferences" }, { v: "journal", l: "Journals" }]} value={kind} onChange={setKind} /></Filters>
      <div className="grid">
        <Card title="Series still active" sub="Grouped by the decade the series started."><Chart option={life} /></Card>
        <Card title="Series lifespan" sub="Median number of years with papers, by starting decade."><Chart option={lifeYears} /></Card>
        <Card span2 title="Publishing spread out, then concentrated again" sub="Share of each year's papers in that year's 10 largest series."><Chart option={conc} height={270} /></Card>
        <Card span2 title="Who publishes it" sub="Share of papers by publisher, identified from DOI prefix. IEEE alone is about a third."><Chart option={pub} height={280} /></Card>
        <Card span2 title="Where the DOI bridge is out" sub="Largest series where under 1% of papers have a DOI — unreachable through OpenAlex. Note ICLR, JMLR and TMLR."><Chart option={doi} height={360} /></Card>
      </div>
    </>
  );
}

// ============================================================= Quality ====
export function PageQuality() {
  const p = usePalette();
  const [rowsOn, setRowsOn] = useState(Object.fromEntries(D.field_coverage.fields.map((f) => [f, true])));
  const rows = D.field_coverage.fields.map((f, i) => ({ f, i })).filter((x) => rowsOn[x.f]);
  const pages = useMemo(() => hbarOption(p, { labels: D.page_formats.map((r) => r.page_format), fmtX: fmt.comma, labelWidth: 170,
    series: [{ name: "Papers", data: D.page_formats.map((r) => r.papers), color: p.s1 }] }), [p]);
  const thesis = useMemo(() => hbarOption(p, { labels: D.thesis_countries.map((r) => r.country), fmtX: fmt.pct1, labelWidth: 90,
    series: [{ name: "Share of theses", data: D.thesis_countries.map((r) => r.pct), color: p.s3 }] }), [p]);
  const rd = useMemo(() => barOption(p, { labels: D.research_data.map((r) => r.year), fmtY: fmt.comma,
    series: [{ name: "Records", data: D.research_data.map((r) => r.records), color: p.s6 }] }), [p]);
  return (
    <>
      <PageHead eyebrow="06 · Records and data quality" title="Every record type has its own schema">
        A conference paper never has a journal; a thesis always has a school. "Missingness" only means something computed per type.
      </PageHead>
      <div className="grid">
        <Card span2 title="Field coverage by record type" sub="Share of records of each type that carry the field. Toggle fields below.">
          <Filters><FilterLabel>Fields</FilterLabel><ChipGroup>
            {D.field_coverage.fields.map((f) => <Chip key={f} label={f} on={rowsOn[f]} onClick={() => setRowsOn({ ...rowsOn, [f]: !rowsOn[f] })} />)}
          </ChipGroup></Filters>
          <Heatmap colLabels={D.field_coverage.types} rowLabels={rows.map((x) => x.f)} matrix={rows.map((x) => D.field_coverage.matrix[x.i])} />
        </Card>
        <Card title="The pages field is free text" sub="Format actually used across all journal and conference papers."><Chart option={pages} height={260} /></Card>
        <Card title="PhD theses by country" sub="Follows which national libraries dblp harvests, not where research happens."><Chart option={thesis} height={320} /></Card>
        <Card span2 title="Research data records" sub="The newest record type — grew 12-fold since 2018. Mostly Zenodo and IEEE DataPort."><Chart option={rd} height={240} /></Card>
      </div>
    </>
  );
}

// ========================================================== Enrichment ====
export function PageEnrichment() {
  const p = usePalette();
  const cov = useMemo(() => hbarOption(p, { labels: ["Found in OpenAlex", "Has an abstract", "Has an institution"], fmtX: fmt.pct1, labelWidth: 130,
    series: D.openalex_coverage.map((r, i) => ({ name: cap(r.kind), color: [p.s1, p.s2, p.s3][i], data: [r.found, r.abstract, r.institution] })) }), [p]);
  const fields = useMemo(() => hbarOption(p, { labels: D.openalex_fields.map((r) => r.field), fmtX: fmt.pct1, labelWidth: 200,
    series: [{ name: "Share of papers", data: D.openalex_fields.map((r) => r.pct), color: p.s4 }] }), [p]);
  return (
    <>
      <PageHead eyebrow="07 · Checked against OpenAlex" title="How good is the DOI bridge?">
        700 random DOIs per kind, 2000–2025, looked up in OpenAlex — the open catalogue with the abstracts, citations and institutions dblp lacks. Each figure is accurate to about ±4 points.
      </PageHead>
      <div className="grid">
        <Card title="What linking through DOIs would add" sub="Abstract and institution shares are for 2013–2025 papers."><Chart option={cov} height={280} /></Card>
        <Card title="Only 51% is primarily computer science" sub="OpenAlex's primary field classification of dblp papers."><Chart option={fields} height={320} /></Card>
      </div>
      <Callout><b>dblp's scope is "publishes in CS venues,"</b> a much wider net than "is CS research." The largest single conference in dblp, ICASSP, is a signal-processing meeting.</Callout>
    </>
  );
}

// =============================================================== Tails ====
export function PageTails() {
  const p = usePalette();
  const panels = Object.keys(D.heavy_tails);
  const [panel, setPanel] = useState(panels[0]);
  const colors = { "Papers per author": p.s1, "Co-authors per author": p.s2, "Papers per venue series": p.s3 };
  const opt = useMemo(() => loglogOption(p, { name: panel, color: colors[panel] || p.s1, data: D.heavy_tails[panel], xLabel: panel }), [p, panel]);
  const stats = {
    "Papers per author": { n: "4,151,660", median: "2", max: "3,351", gini: "0.727", top1: "26.6%", alpha: "2.14 ± 0.01 (from 13)" },
    "Co-authors per author": { n: "4,010,661", median: "5", max: "2,954", gini: "0.638", top1: "16.8%", alpha: "3.00 ± 0.02 (from 84)" },
    "Papers per venue series": { n: "8,490", median: "179", max: "110,888", gini: "0.779", top1: "24.0%", alpha: "2.38 ± 0.06 (from 2,583)" },
  }[panel];
  return (
    <>
      <PageHead eyebrow="08 · The long tail, formally" title="Heavy-tailed, but not power laws">
        The top 1% of authors hold 27% of all author slots. But a lognormal fits every one of these distributions better than a pure power law.
      </PageHead>
      <Filters page><FilterLabel>Distribution</FilterLabel><Seg options={panels.map((x) => ({ v: x, l: x }))} value={panel} onChange={setPanel} /></Filters>
      <KpiStrip items={[
        { n: stats.n, l: "entities" }, { n: stats.median, l: "median" }, { n: stats.max, l: "maximum" },
        { n: stats.gini, l: "Gini coefficient" }, { n: stats.top1, l: "held by the top 1%" }, { n: stats.alpha, l: "power-law exponent α in the tail" },
      ]} />
      <div className="grid">
        <Card span2 title={`Share with at least x: ${panel.toLowerCase()}`} sub="Log–log. A straight line would be a power law; the curve bends down instead. Disambiguation bins excluded.">
          <Chart option={opt} height={380} />
        </Card>
      </div>
      <Callout>Fit with the Clauset–Shalizi–Newman procedure: papers-per-author α = 2.14 in the tail, close to Lotka's classic α ≈ 2 — but a lognormal beats the power law in every likelihood-ratio test (p &lt; 0.05). "Heavy-tailed, not scale-free."</Callout>
    </>
  );
}
