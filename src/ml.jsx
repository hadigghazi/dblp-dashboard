import { useState } from "react";
import { useApi, useDebounced, fmtDate } from "./api.js";
import { fmt } from "./charts.jsx";
import {
  KpiStrip, PageHead, Card, Filters, FilterLabel, SearchBox, SortableTable, KindBadge, Callout, EmptyNote, Spinner,
} from "./components.jsx";

const pct = (v, d = 1) => (v == null ? "—" : `${(100 * v).toFixed(d)}%`);
const num = (v, d = 3) => (v == null ? "—" : Number(v).toFixed(d));

/** The model's measured accuracy, shown wherever its output is - a suggestion never travels alone. */
export function ModelCard({ card, compact }) {
  if (!card) return null;
  if (!card.available) {
    return <Callout><b>No trained model yet.</b> {card.error} On the VM: <code>docker compose -f docker-compose.prod.yml run --rm ml python -m ml.cli train</code></Callout>;
  }
  const a = card.assignment || {};
  const c = card.clustering || {};
  const b = card.clustering_bin_like || {};
  const items = [
    { n: pct(a.top1_accuracy), l: `puts a hidden paper on the right person (co-author rule: ${pct(a.top1_accuracy_overlap_baseline)})` },
    { n: pct(a.precision_when_shown), l: `right when it names someone, on ${pct(a.coverage, 0)} of cases; silent otherwise` },
    { n: num(c.b3_f1), l: `B³ F1 splitting labelled name blocks (rule: ${num(c.b3_f1_overlap_baseline)})` },
    { n: num(b.b3_f1), l: `B³ F1 on bin-like data, 1–2 papers per person (doing nothing: ${num(b.b3_f1_all_singletons)}, rule: ${num(b.b3_f1_overlap_baseline)})` },
  ];
  if (compact) {
    return (
      <div className="modelnote">
        <span className="srctag job">Model · trained {fmtDate(card.trained_at)}</span>
        {" "}Right {pct(a.precision_when_shown, 0)} of the time when it names a person; correct person in the top spot {pct(a.top1_accuracy, 0)}{" "}
        (held-out name blocks, vs {pct(a.top1_accuracy_overlap_baseline, 0)} for “share a co-author”).
      </div>
    );
  }
  return (
    <>
      <KpiStrip items={items} />
      <div className="grid">
        <Card title="What carries the signal" sub="Permutation importance: how much average precision drops when the feature is shuffled.">
          <SortableTable rows={card.feature_importance || []} defaultSort={{ key: "drop_in_average_precision", dir: -1 }} columns={[
            { key: "feature", label: "Feature", render: (r) => <code>{r.feature}</code> },
            { key: "drop_in_average_precision", label: "Drop in AP", num: true, render: (r) => num(r.drop_in_average_precision, 3) },
          ]} />
        </Card>
        <Card title="Training data" sub="Labels are dblp's numbered pages: each is a person an editor verified. Split by whole name block.">
          <SortableTable rows={[
            { k: "Pairs of papers", v: fmt.comma(card.dataset?.pairs || 0) },
            { k: "of which same person", v: fmt.comma(card.dataset?.positives || 0) },
            { k: "Name blocks", v: fmt.comma(card.dataset?.blocks || 0) },
            { k: "Held-out test pairs", v: fmt.comma(card.dataset?.test || 0) },
            { k: "Pairwise ROC-AUC (test)", v: num(card.pairwise?.roc_auc) + ` vs rule ${num(card.pairwise_overlap_baseline?.roc_auc)}` },
            { k: "Dump", v: `${card.dump?.fingerprint} · ${fmt.comma(Number(card.dump?.records || 0))} records` },
            { k: "Trained", v: fmtDate(card.trained_at) },
          ]} defaultSort={{ key: "k", dir: 1 }} columns={[{ key: "k", label: "" }, { key: "v", label: "" }]} />
        </Card>
      </div>
    </>
  );
}

function ClusterRow({ c, go }) {
  const s = c.suggested_person;
  const u = c.best_candidate_below_threshold;
  let head;
  if (s) {
    head = (
      <div className="clusterhead named">
        <span className="flag good">looks like</span>
        <button type="button" className="linkish strong" onClick={() => go("authors", { key: s.key })}>{s.name}</button>
        <span className="muted">score {num(s.score, 2)}{s.weakest_member_score != null ? ` (weakest ${num(s.weakest_member_score, 2)}, ${c.merged_from} groups)` : ""}{s.z != null ? ` · z ${s.z}` : ""}</span>
      </div>
    );
  } else if (u) {
    head = (
      <div className="clusterhead">
        <span className="flag">uncertain</span>
        <span>closest: <button type="button" className="linkish" onClick={() => go("authors", { key: u.key })}>{u.name}</button></span>
        <span className="muted">score {num(u.score, 2)}, margin {num(u.margin, 2)} — not enough to name</span>
      </div>
    );
  } else {
    head = <div className="clusterhead"><span className="flag bad">no page yet</span><span className="muted">no numbered page comes close: probably a person dblp hasn’t registered</span></div>;
  }
  return (
    <li className="cluster">
      {head}
      <ul className="paperlist compact">
        {c.papers.map((p) => (
          <li key={p.key}><button type="button" className="paperrow" onClick={() => go("papers", { key: p.key })}>
            <span className="ptitle">{p.title}</span>
            <span className="pmeta">{p.year ?? "—"} · {p.venue || <span className="muted">no venue</span>}</span>
          </button></li>
        ))}
      </ul>
    </li>
  );
}

/** The proposed split of one disambiguation bin. */
export function BinSplit({ authorKey, go }) {
  const [refresh, setRefresh] = useState(0);
  const split = useApi("ml/bin", { key: authorKey, refresh: refresh ? "true" : undefined });
  const status = useApi("ml/status");
  const d = split.data;
  return (
    <Card span2 title="Proposed split" state={split} job={null}
          sub={d ? `${d.papers} of this bin’s papers examined${d.papers < 300 ? "" : " (capped)"} · ${d.numbered_people_in_block} numbered pages in this name block · ${d.cached ? "from cache" : `computed in ${d.computed_in_seconds}s`}`
                 : "Clustering this bin’s papers and matching each group to the numbered pages that share the name. First run on a large bin can take up to a minute."}>
      {(data) => (
        <>
          <ModelCard card={status.data} compact />
          <KpiStrip items={[
            { n: fmt.comma(data.summary.clusters), l: "groups of papers that look like one person each" },
            { n: fmt.comma(data.summary.matched_to_a_numbered_page), l: `groups confidently matched to an existing page (${fmt.comma(data.summary.papers_matched)} papers)` },
            { n: fmt.comma(data.summary.uncertain), l: "groups with a plausible but unproven candidate" },
            { n: fmt.comma(data.summary.look_new), l: "groups that match no page: people with no entry yet" },
          ]} />
          {data.summary.pages_named_by_several_groups?.length ? (
            <Callout>{data.summary.pages_named_by_several_groups.length} numbered page(s) are named by more than one group that couldn’t be merged confidently — either one person split in two, or a page that itself mixes people. Worth a human look.</Callout>
          ) : null}
          <ol className="clusterlist">
            {data.clusters.map((c, i) => <ClusterRow key={i} c={c} go={go} />)}
          </ol>
          <div className="filters"><button type="button" className="btn" onClick={() => setRefresh(refresh + 1)}>Recompute</button>
            <span className="hint">Thresholds: cluster {num(data.thresholds.cluster, 2)}, name {num(data.thresholds.assign, 2)}, margin {num(data.thresholds.min_margin, 2)}, outlier z ≥ {data.thresholds.min_z}, strongest link ≥ {num(data.thresholds.min_max_link, 2)}</span></div>
        </>
      )}
    </Card>
  );
}

/** The Disambiguation page: the model card and the biggest bins to try it on. */
export function PageDisambiguation({ params, go }) {
  const status = useApi("ml/status");
  const [q, setQ] = useState(params.q || "");
  const qd = useDebounced(q, 400);
  const bins = useApi("ml/bins", { top: 40, q: qd.trim() || undefined });
  return (
    <>
      <PageHead eyebrow="Machine learning · Author disambiguation" title="Who is this “Wei Wang”?">
        dblp records authors as name strings. When editors haven’t decided which person a paper belongs to, it sits in a
        disambiguation bin with everyone else of that name. A model trained on the 147,223 pages editors <em>have</em> verified proposes
        how to split a bin — and says how sure it is.
      </PageHead>
      <Card state={status} title="How good is it" sub="Every number is measured on name blocks the model never saw in training, next to the rule a person would write by hand: “same person if they share a co-author”.">
        {(card) => <ModelCard card={card} />}
      </Card>
      <div style={{ height: 18 }} />
      <Card state={bins} title="The biggest bins" sub="Papers waiting to be assigned, by bin. Click one to see the proposed split.">
        {(d) => (
          <>
            <Filters><SearchBox id="bin-q" value={q} onChange={setQ} placeholder="Filter by name, e.g. Wei Wang" /></Filters>
            {d.bins.length ? (
              <SortableTable rows={d.bins} defaultSort={{ key: "papers", dir: -1 }} onRowClick={(r) => go("authors", { key: r.key })} columns={[
                { key: "name", label: "Bin", render: (r) => <><span className="strong">{r.name}</span> <KindBadge kind="disambiguation" /></> },
                { key: "papers", label: "Unassigned papers", num: true, render: (r) => fmt.comma(r.papers) },
                { key: "numbered_pages", label: "Numbered pages with this name", num: true, render: (r) => fmt.comma(r.numbered_pages) },
              ]} />
            ) : <EmptyNote>No bin matches “{qd}”.</EmptyNote>}
          </>
        )}
      </Card>
    </>
  );
}
