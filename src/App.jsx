import { useCallback, useEffect, useState } from "react";
import { PageOverview, PagePublishing, PageIdentity, PageNetwork, PageTitles, PageVenues, PageQuality, PageEnrichment, PageTails } from "./pages.jsx";
import { PageAuthors, PageVenueExplorer, PagePapers } from "./explore.jsx";
import { PageDisambiguation, PageCollaborators, PageWhereToPublish } from "./ml.jsx";
import { useStatus } from "./api.js";

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
  { id: "disambiguation", no: "M1", label: "Disambiguation", group: "Machine learning", Comp: PageDisambiguation },
  { id: "collaborators", no: "M2", label: "Next co-authors", group: "Machine learning", Comp: PageCollaborators },
  { id: "where-to-publish", no: "M3", label: "Where to publish", group: "Machine learning", Comp: PageWhereToPublish },
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
          <button className="themebtn" onClick={cycleTheme}>Theme: {theme}</button>
        </div>
      </aside>
      <main>
        <current.Comp key={current.id} params={route.params} go={go} status={status} />
      </main>
    </>
  );
}
