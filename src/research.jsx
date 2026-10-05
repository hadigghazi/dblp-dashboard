import { useMemo } from "react";
import { useStatic } from "./api.js";
import { Chart, usePalette, recoveryOption } from "./charts.jsx";
import { KpiStrip, PageHead, Card, EmptyNote } from "./components.jsx";

// Everything on this page comes from paper.json, which the Paper workflow publishes with the PDF: the
// title and abstract are read from the paper's LaTeX source, the rest from site.json beside it. So the
// page changes when the paper does, without a new build of the site.
const BASE = "/downloads/research/dblpqa/";
const MODELS = ["gpt-4.1", "gpt-4.1-mini", "Mistral-7B"];

function size(n) {
  if (!n) return "";
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.round(n / 1e3)} kB`;
}

function day(iso) {
  return iso ? new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" }) : "";
}

function PaperCard({ paper }) {
  const facts = [paper.pages ? `${paper.pages} pages` : null, paper.updated ? `updated ${day(paper.updated)}` : null,
                 paper.commit ? `version ${paper.commit}` : null].filter(Boolean).join(" · ");
  return (
    <Card span2 title="The paper" sub={facts}>
      <h4 className="papertitle">{paper.title}</h4>
      <p className="paperby">{paper.author}{paper.affiliation ? ` · ${paper.affiliation}` : ""}</p>
      <p className="abstract">{paper.abstract}</p>
      <div className="paperactions">
        <a className="btn primary" href={BASE + paper.pdf} target="_blank" rel="noopener">
          Read the paper (PDF{paper.bytes ? `, ${size(paper.bytes)}` : ""})
        </a>
        <a className="btn ghost" href={BASE + paper.pdf} download>Download</a>
        <a className="btn ghost" href={BASE + paper.bib} download>Cite (BibTeX)</a>
      </div>
    </Card>
  );
}

export function PageResearch({ ask }) {
  const p = usePalette();
  const doc = useStatic(`${BASE}paper.json`);
  const d = doc.data;
  const recovery = useMemo(() => d?.recovery && recoveryOption(p, {
    xName: "questions with the right paper in the context",
    yName: "share of the gain recovered",
    series: MODELS.map((name, i) => ({ name, color: [p.s1, p.s2, p.s3][i],
                                       data: d.recovery.points.filter((pt) => pt.model === name) })),
  }), [p, d]);

  const head = (
    <PageHead eyebrow="R1 · Research · Dewey" title={d?.page_title || "What does retrieval add?"}>
      {d?.lede || "A study of retrieval-augmented question answering over dblp, and the paper that reports it."}
    </PageHead>
  );
  if (doc.missing || doc.error) {
    return (
      <div className="research">
        {head}
        <section className="card">
          {doc.missing ? <EmptyNote>The paper hasn’t been published on this server yet.</EmptyNote>
            : <div className="cardmsg error" role="alert"><b>Couldn’t load the research page.</b> {doc.error}</div>}
        </section>
      </div>
    );
  }

  return (
    <div className="research">
      {head}
      <KpiStrip items={d?.kpis} loading={!d} />
      {!d ? <div className="skel" style={{ height: 320, marginTop: 18 }} /> : (
        <div className="grid">
          <PaperCard paper={d.paper} />

          <Card span2 title="What we found">
            <ol className="findings">
              {d.findings.map((f) => <li key={f.title}><b>{f.title}.</b> {f.text}</li>)}
            </ol>
          </Card>

          <Card title="The gain follows retrieval success" sub={d.recovery.caption} height={320}>
            <Chart option={recovery} height={320}
                   label="Share of the right abstract's gain recovered by RAG, against how often the right paper is retrieved, per model" />
          </Card>

          <Card title="Scores" sub={d.results.caption}>
            <div className="tablewrap">
              <table className="data">
                <thead><tr>{d.results.columns.map((c, i) => <th key={c} style={i > 1 ? { textAlign: "right" } : null}>{c}</th>)}</tr></thead>
                <tbody>
                  {d.results.rows.map((r) => (
                    <tr key={r[0] + r[1]}>{r.map((v, i) => <td key={i} className={i > 1 ? "num" : undefined}>{v}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card span2 title="What it means for Dewey">
            {d.dewey.map((t, i) => <p key={i} className="dlsum">{t}</p>)}
            {ask ? <button type="button" className="btn ghost" onClick={ask}>Ask Dewey</button> : null}
          </Card>

          <Card span2 title="Downloads and sources">
            <ul className="srclist">
              <li><a href={BASE + d.paper.pdf} download>The paper</a> (PDF{d.paper.pages ? `, ${d.paper.pages} pages` : ""})</li>
              <li><a href={BASE + d.paper.bib} download>BibTeX entry</a></li>
              <li><a href={`${BASE}dblpqa-fresh.csv`} download>DBLP-QA-Fresh</a>: the 100 questions about 2025–2026
                papers, with their reference answers and dblp keys (CSV), built automatically for the study</li>
              {d.links.map((l) => <li key={l.href}><a href={l.href} target="_blank" rel="noopener">{l.label}</a></li>)}
            </ul>
          </Card>
        </div>
      )}
    </div>
  );
}
