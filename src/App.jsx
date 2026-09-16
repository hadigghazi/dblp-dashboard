import { useEffect, useState } from "react";
import { PageOverview, PagePublishing, PageIdentity, PageNetwork, PageTitles, PageVenues, PageQuality, PageEnrichment, PageTails } from "./pages.jsx";

const PAGES = [
  { id: "overview", no: "–", label: "Overview", group: null, Comp: PageOverview },
  { id: "publishing", no: "01", label: "Publishing", group: "Findings", Comp: PagePublishing },
  { id: "identity", no: "02", label: "Author identity", group: "Findings", Comp: PageIdentity },
  { id: "network", no: "03", label: "Co-author network", group: "Findings", Comp: PageNetwork },
  { id: "titles", no: "04", label: "Titles & topics", group: "Findings", Comp: PageTitles },
  { id: "venues", no: "05", label: "Venues & publishers", group: "Findings", Comp: PageVenues },
  { id: "quality", no: "06", label: "Records & quality", group: "Findings", Comp: PageQuality },
  { id: "enrichment", no: "07", label: "OpenAlex check", group: "Findings", Comp: PageEnrichment },
  { id: "tails", no: "08", label: "Long tails", group: "Findings", Comp: PageTails },
];

function readHash() {
  const id = location.hash.replace("#", "");
  return PAGES.some((p) => p.id === id) ? id : "overview";
}

function useTheme() {
  const [theme, setTheme] = useState(() => localStorage.getItem("dblp-theme") || "system");
  useEffect(() => {
    if (theme === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("dblp-theme", theme);
  }, [theme]);
  const next = { system: "light", light: "dark", dark: "system" };
  return [theme, () => setTheme(next[theme])];
}

export default function App() {
  const [page, setPage] = useState(readHash);
  const [navOpen, setNavOpen] = useState(false);
  const [theme, cycleTheme] = useTheme();

  useEffect(() => {
    const onHash = () => setPage(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const go = (id) => { location.hash = id; setPage(id); setNavOpen(false); window.scrollTo(0, 0); };

  const current = PAGES.find((p) => p.id === page) || PAGES[0];
  let lastGroup = null;

  return (
    <>
      <button className="menubtn" onClick={() => setNavOpen(!navOpen)} aria-label="Toggle navigation">{"☰"}</button>
      <aside className={"sidebar" + (navOpen ? " open" : "")}>
        <div className="brand">
          <div className="mark"><b>dblp</b> / explorer</div>
          <h1>dblp Explorer</h1>
          <div className="sub">12.9M records, filtered live</div>
        </div>
        <nav className="pages" aria-label="Sections">
          {PAGES.map((p) => {
            const showGroup = p.group && p.group !== lastGroup;
            lastGroup = p.group;
            return (
              <div key={p.id}>
                {showGroup ? <div className="navsec">{p.group}</div> : null}
                <button className={"navbtn" + (page === p.id ? " active" : "")} onClick={() => go(p.id)} style={{ width: "100%" }}>
                  <span className="no">{p.no}</span>{p.label}
                </button>
              </div>
            );
          })}
        </nav>
        <div className="sidebar-foot">
          Source: dblp.xml.gz, 2026-09-01 · DOI 10.4230/dblp.xml.2026-09-01 · CC0
          <div><button className="themebtn" onClick={cycleTheme}>Theme: {theme}</button></div>
        </div>
      </aside>
      <main>
        <current.Comp />
      </main>
    </>
  );
}
