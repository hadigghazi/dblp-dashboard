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

// ============================================================== Link prediction ====

const HEURISTIC_NAMES = { cn: "Common neighbours", jaccard: "Jaccard", aa: "Adamic–Adar", ra: "Resource allocation", pa: "Preferential attachment" };

/** The link model's measured accuracy, next to the classic graph heuristics on the same candidates. */
export function LinksModelCard({ card, compact }) {
  if (!card) return null;
  if (!card.available) {
    return <Callout><b>No trained link model yet.</b> {card.error} On the VM: <code>docker compose -f docker-compose.prod.yml run --rm ml python -m ml.links.cli train</code></Callout>;
  }
  const t = card.test || {};
  const r = t.ranking || {};
  const p = t.pooled || {};
  const b = card.baselines || {};
  const bestOf = (key, pick) => {
    let best = null;
    for (const [h, m] of Object.entries(b)) {
      const v = pick(m);
      if (v != null && (best == null || v > best.v)) best = { h, v };
    }
    return best;
  };
  const bestMrr = bestOf("mrr", (m) => m.ranking?.mrr);
  const bestHits = bestOf("hits", (m) => m.ranking?.["hits@10"]);
  const bestAuc = bestOf("auc", (m) => m.pooled?.roc_auc);
  const origin = card.dataset?.test?.new_links_origin || {};
  if (compact) {
    return (
      <div className="modelnote">
        <span className="srctag job">Model · trained {fmtDate(card.trained_at)}</span>
        {" "}Tested on the {t.snapshot} snapshot, predicting {t.horizon} years ahead for authors it never saw: the right person is in the top 10 for{" "}
        {pct(r["hits@10"], 0)} of authors who gained a new distance-2 co-author (best heuristic, {HEURISTIC_NAMES[bestHits?.h] || "—"}: {pct(bestHits?.v, 0)}).
      </div>
    );
  }
  const heurRows = Object.entries(b).map(([h, m]) => ({
    heuristic: HEURISTIC_NAMES[h] || h, roc_auc: m.pooled?.roc_auc, mrr: m.ranking?.mrr, hits10: m.ranking?.["hits@10"], p5: m.ranking?.["precision@5"],
  }));
  heurRows.unshift({ heuristic: "Model", roc_auc: p.roc_auc, mrr: r.mrr, hits10: r["hits@10"], p5: r["precision@5"], model: true });
  return (
    <>
      <KpiStrip items={[
        { n: pct(r["hits@10"], 0), l: `authors with the right new co-author in their top 10 (best heuristic, ${HEURISTIC_NAMES[bestHits?.h] || "—"}: ${pct(bestHits?.v, 0)})` },
        { n: num(r.mrr, 2), l: `mean reciprocal rank of the first correct suggestion (${HEURISTIC_NAMES[bestMrr?.h] || "—"}: ${num(bestMrr?.v, 2)})` },
        { n: num(p.roc_auc, 3), l: `ROC-AUC over all candidate pairs (${HEURISTIC_NAMES[bestAuc?.h] || "—"}: ${num(bestAuc?.v, 3)})` },
        { n: pct(origin["distance 2"]?.share, 0), l: "of new co-authors were a co-author of a co-author at the snapshot: the ceiling for any local method" },
      ]} />
      <div className="grid">
        <Card title="Model versus the classic heuristics" sub={`Same candidate pairs, snapshot ${t.snapshot} → ${t.snapshot + (t.horizon || 0)}; ranking metrics over ${fmt.comma(r.anchors_with_new_link || 0)} authors who gained a distance-2 co-author.`}>
          <SortableTable rows={heurRows} defaultSort={{ key: "mrr", dir: -1 }} columns={[
            { key: "heuristic", label: "Ranker", render: (x) => (x.model ? <span className="strong">{x.heuristic}</span> : x.heuristic) },
            { key: "roc_auc", label: "ROC-AUC", num: true, render: (x) => num(x.roc_auc, 3) },
            { key: "mrr", label: "MRR", num: true, render: (x) => num(x.mrr, 3) },
            { key: "hits10", label: "Hits@10", num: true, render: (x) => pct(x.hits10, 0) },
            { key: "p5", label: "P@5", num: true, render: (x) => pct(x.p5, 1) },
          ]} />
        </Card>
        <Card title="Where new co-authors come from" sub="The test authors' new co-authors, by where they stood at the snapshot. Only the first row is reachable by this method.">
          <SortableTable rows={["distance 2", "farther", "newcomer"].map((k) => ({ k, ...(origin[k] || {}) }))} defaultSort={{ key: "share", dir: -1 }} columns={[
            { key: "k", label: "At the snapshot", render: (x) => ({ "distance 2": "Co-author of a co-author", farther: "Publishing, but farther away", newcomer: "No paper yet (newcomer)" }[x.k]) },
            { key: "links", label: "New links", num: true, render: (x) => fmt.comma(x.links || 0) },
            { key: "share", label: "Share", num: true, render: (x) => pct(x.share, 1) },
          ]} />
        </Card>
        <Card title="What a score means" sub="Pairs of the test snapshot by model score: how many of them did become co-authors within the horizon.">
          <SortableTable rows={card.calibration || []} defaultSort={{ key: "from", dir: 1 }} columns={[
            { key: "from", label: "Score", render: (x) => `${num(x.from, 2)} – ${num(x.to, 2)}` },
            { key: "pairs", label: "Pairs", num: true, render: (x) => fmt.comma(x.pairs) },
            { key: "came_true", label: "Came true", num: true, render: (x) => pct(x.came_true, 1) },
          ]} />
        </Card>
        <Card title="What carries the signal" sub="Permutation importance: how much average precision drops when the feature is shuffled.">
          <SortableTable rows={card.feature_importance || []} defaultSort={{ key: "drop_in_average_precision", dir: -1 }} columns={[
            { key: "feature", label: "Feature", render: (x) => <code>{x.feature}</code> },
            { key: "drop_in_average_precision", label: "Drop in AP", num: true, render: (x) => num(x.drop_in_average_precision, 3) },
          ]} />
        </Card>
        <Card span2 title="Training data" sub="Labels are real: co-author pairs that formed in the years after each snapshot. Authors are split by hash, so nobody is an anchor in both snapshots.">
          <SortableTable rows={[
            { k: "Train snapshot", v: `${card.dataset?.train?.snapshot} → ${card.dataset?.train?.snapshot + (card.dataset?.train?.horizon || 0)} · ${fmt.comma(card.dataset?.train?.anchors || 0)} anchors · ${fmt.comma(card.dataset?.train?.pairs_used || 0)} pairs (${fmt.comma(card.dataset?.train?.positives || 0)} became co-authors)` },
            { k: "Test snapshot", v: `${card.dataset?.test?.snapshot} → ${card.dataset?.test?.snapshot + (card.dataset?.test?.horizon || 0)} · ${fmt.comma(card.dataset?.test?.anchors || 0)} anchors · ${fmt.comma(card.dataset?.test?.candidate_pairs || 0)} candidate pairs (${fmt.comma(card.dataset?.test?.positives || 0)} became co-authors)` },
            { k: "Dump", v: `${card.dump?.fingerprint} · ${fmt.comma(Number(card.dump?.records || 0))} records` },
            { k: "Trained", v: fmtDate(card.trained_at) },
          ]} defaultSort={{ key: "k", dir: 1 }} columns={[{ key: "k", label: "" }, { key: "v", label: "" }]} />
        </Card>
      </div>
    </>
  );
}

/** Likely next co-authors for one author page. */
export function LinkSuggestions({ authorKey, go }) {
  const [refresh, setRefresh] = useState(0);
  const links = useApi("ml/links", { key: authorKey, top: 10, refresh: refresh ? "true" : undefined });
  const status = useApi("ml/links/status");
  const d = links.data;
  return (
    <Card span2 title="Likely next co-authors" state={links} job={null}
          sub={d ? `${fmt.comma(d.candidates)} co-authors of co-authors considered · ${d.cached ? "from cache" : `computed in ${d.computed_in_seconds}s`}`
                 : "Every co-author of a co-author who is not a co-author yet, ranked by the model."}>
      {(data) => (
        <>
          <LinksModelCard card={status.data} compact />
          {data.note ? <EmptyNote>{data.note}.</EmptyNote> : null}
          <ol className="clusterlist">
            {data.suggestions.map((s) => (
              <li key={s.key} className="cluster">
                <div className="clusterhead named">
                  <span className="flag">#{s.rank}</span>
                  <button type="button" className="linkish strong" onClick={() => go("authors", { key: s.key })}>{s.name}</button>
                  <span className="muted">score {num(s.score, 2)}{s.came_true != null ? ` · pairs scored like this came true ${pct(s.came_true, 0)} of the time` : ""}</span>
                </div>
                <div className="pmeta" style={{ padding: "6px 0 8px" }}>
                  <span>{s.common_coauthors} shared co-author{s.common_coauthors === 1 ? "" : "s"}{s.via.length ? ": " : ""}</span>
                  {s.via.map((w, i) => (
                    <span key={w.key}>
                      <button type="button" className="linkish" onClick={() => go("authors", { key: w.key })}>{w.name}</button>{i < s.via.length - 1 ? "," : ""}
                    </span>
                  ))}
                  {s.shared_venues.length ? <span>· shared venues: {s.shared_venues.map((v) => v.name).join(", ")}</span> : null}
                  <span>· {fmt.comma(s.papers || 0)} papers, last {s.last_year ?? "—"}</span>
                </div>
              </li>
            ))}
          </ol>
          <div className="filters"><button type="button" className="btn" onClick={() => setRefresh(refresh + 1)}>Recompute</button>
            <span className="hint">Candidates are co-authors of co-authors only; people farther away or not yet publishing cannot be suggested.</span></div>
        </>
      )}
    </Card>
  );
}

/** The Collaborators page: the model card and an author to try it on. */
export function PageCollaborators({ params, go }) {
  const status = useApi("ml/links/status");
  const [q, setQ] = useState(params.q || "");
  const qd = useDebounced(q, 400);
  const search = useApi(qd.trim().length >= 2 ? "authors/search" : null, { q: qd.trim() });
  const key = params.key;
  return (
    <>
      <PageHead eyebrow="Machine learning · Link prediction" title="Who will they write with next?">
        New collaborations mostly close triangles: a co-author of a co-author becomes a co-author. A model trained on the
        co-authorship graph as it stood in one year, and on who actually collaborated in the two years after, ranks an author’s
        distance-2 neighbours — and says how often a score like that has come true.
      </PageHead>
      <Card state={status} title="How good is it" sub="Measured on a later snapshot than the one it was trained on, for authors it never saw, next to the classic heuristics of the link-prediction literature.">
        {(card) => <LinksModelCard card={card} />}
      </Card>
      <div style={{ height: 18 }} />
      <Card title="Try an author" sub="Search a name, pick a page: the suggestions also appear on every author page.">
        <Filters><SearchBox id="links-q" value={q} onChange={setQ} placeholder="Search a name, e.g. Yoshua Bengio" /></Filters>
        {search.data && !key ? (
          search.data.length ? (
            <SortableTable rows={search.data.filter((r) => r.page_kind !== "disambiguation")} defaultSort={{ key: "papers", dir: -1 }} onRowClick={(r) => go("collaborators", { q: qd, key: r.key })} columns={[
              { key: "name", label: "Name", render: (r) => <><span className="strong">{r.name}</span> {r.page_kind !== "regular" ? <KindBadge kind={r.page_kind} /> : null}</> },
              { key: "papers", label: "Papers", num: true, render: (r) => fmt.comma(r.papers) },
              { key: "first_year", label: "Active", render: (r) => (r.first_year ? `${r.first_year}–${r.last_year}` : "—") },
            ]} />
          ) : <EmptyNote>No author page matches “{qd}”.</EmptyNote>
        ) : null}
        {search.loading && !key ? <Spinner /> : null}
      </Card>
      {key ? (
        <>
          <div style={{ height: 18 }} />
          <div className="grid"><LinkSuggestions authorKey={key} go={go} /></div>
          <div className="filters"><button type="button" className="linkish" onClick={() => go("collaborators", { q: params.q })}>&larr; Pick another author</button></div>
        </>
      ) : null}
    </>
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
