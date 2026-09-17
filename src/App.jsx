import { useCallback, useEffect, useState } from "react";
import { PageOverview, PagePublishing, PageIdentity, PageNetwork, PageTitles, PageVenues, PageQuality, PageEnrichment, PageTails } from "./pages.jsx";
import { PageAuthors, PageVenueExplorer, PagePapers } from "./explore.jsx";
import { useStatus, fmtDate } from "./api.js";
import { Spinner } from "./components.jsx";

const PAGES = [
  { id: "overview", no: "–", label: "Overview", group: null, Comp: PageOverview },
  { id: "publishing", no: "01", label: "Publishing", group: "Findings", Comp: PagePublishing },
  { id: "identity", no: "02", label: "Author identity", group: "Findings", Comp: PageIdentity },
  { id: "network", no: "03", label: "Co-author network", group: "Findings", Comp: PageNetwork },
  { id: "titles", no: "04", label: "Titles & topics", group: "Findings", Comp: PageTitles },
  { id: "venue-trends", no: "05", label: "Venues & publishers", group: "Findings", Comp: PageVenues },
  { id: "quality", no: "06", label: "Records & quality", group: "Findings", Comp: PageQuality },
  { id: "enrichment", no: "07", label: "OpenAlex check", group: "Findings", Comp: PageEnrichment },
  { id: "tails", no: "08", label: "Long tails", group: "Findings", Comp: PageTails },
  { id: "authors", no: "A", label: "Authors", group: "Explore", Comp: PageAuthors },
  { id: "venues", no: "V", label: "Venues", group: "Explore", Comp: PageVenueExplorer },
  { id: "papers", no: "P", label: "Papers", group: "Explore", Comp: PagePapers },
];

/** "#authors?key=homepages/1/2" -> { page: "authors", params: { key: "homepages/1/2" } } */
function readRoute() {
  const raw = location.hash.replace(/^#/, "");
  const [id, qs] = raw.split("?");
  const page = PAGES.some((p) => p.id === id) ? id : "overview";
  return { page, params: Object.fromEntries(new URLSearchParams(qs || "")) };
}

function routeHash(page, params) {
  const clean = Object.fromEntries(Object.entries(params || {}).filter(([, v]) => v !== undefined && v !== null && v !== ""));
  const qs = new URLSearchParams(clean).toString();
  return `#${page}${qs ? `?${qs}` : ""}`;
}

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("dblp-theme") || "system"; } catch { return "system"; }
  });
  useEffect(() => {
    if (theme === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("dblp-theme", theme); } catch { /* private mode */ }
  }, [theme]);
  const next = { system: "light", light: "dark", dark: "system" };
  return [theme, () => setTheme(next[theme])];
}

function StatusBlock({ status }) {
  const s = status?.status;
  if (!s) return <div className="status"><Spinner /> Connecting…</div>;
  if (s.state === "ready") {
    const m = status.meta || {};
    return (
      <div className="status">
        <div><span className="dot live" aria-hidden="true" /><b>Live</b> · {Number(m.records || 0).toLocaleString()} records</div>
        <div>Latest record edit {fmtDate(m.latest_mdate)}</div>
        <div>Tables built {fmtDate(m.built_at)}</div>
        {s.refreshing ? <div className="warn"><Spinner /> New dump found, rebuilding: {s.refreshing}</div> : null}
        {s.optional_failed ? <div className="warn">Unavailable: {s.optional_failed}</div> : null}
      </div>
    );
  }
  if (s.state === "building" || s.state === "starting") {
    return (
      <div className="status">
        <div><Spinner /> <b>Preparing live data</b></div>
        <div>{s.message} ({s.step}/{s.steps})</div>
        <div className="progress"><span style={{ width: `${Math.round((100 * (s.step || 0)) / (s.steps || 1))}%` }} /></div>
        <div className="muted">Only after a new dump: this takes a few minutes.</div>
      </div>
    );
  }
  return <div className="status"><div><span className="dot off" aria-hidden="true" /><b>Unavailable</b></div><div>{s.message}</div></div>;
}

export default function App() {
  const [route, setRoute] = useState(readRoute);
  const [navOpen, setNavOpen] = useState(false);
  const [theme, cycleTheme] = useTheme();
  const status = useStatus();

  useEffect(() => {
    const onHash = () => setRoute(readRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = useCallback((page, params = {}, { replace = false } = {}) => {
    const hash = routeHash(page, params);
    if (hash === location.hash) return;
    if (replace) {
      history.replaceState(null, "", hash);
    } else {
      history.pushState(null, "", hash);
      window.scrollTo(0, 0);
    }
    setRoute(readRoute());
    setNavOpen(false);
  }, []);

  const current = PAGES.find((p) => p.id === route.page) || PAGES[0];
  let lastGroup = null;

  return (
    <>
      <button className="menubtn" onClick={() => setNavOpen(!navOpen)} aria-label="Toggle navigation" aria-expanded={navOpen}>{"☰"}</button>
      <aside className={"sidebar" + (navOpen ? " open" : "")}>
        <div className="brand">
          <div className="mark"><b>dblp</b> / explorer</div>
          <h1>dblp Explorer</h1>
          <div className="sub">Queried live from the dump</div>
        </div>
        <nav className="pages" aria-label="Sections">
          {PAGES.map((p) => {
            const showGroup = p.group && p.group !== lastGroup;
            lastGroup = p.group;
            return (
              <div key={p.id}>
                {showGroup ? <div className="navsec">{p.group}</div> : null}
                <a className={"navbtn" + (route.page === p.id ? " active" : "")} href={`#${p.id}`}
                   aria-current={route.page === p.id ? "page" : undefined}
                   onClick={(e) => { e.preventDefault(); go(p.id, {}); }}>
                  <span className="no">{p.no}</span>{p.label}
                </a>
              </div>
            );
          })}
        </nav>
        <div className="sidebar-foot">
          <StatusBlock status={status} />
          <div className="footline">Source: dblp.xml.gz · CC0</div>
          <button className="themebtn" onClick={cycleTheme}>Theme: {theme}</button>
        </div>
      </aside>
      <main>
        <current.Comp key={current.id} params={route.params} go={go} status={status} />
      </main>
    </>
  );
}
