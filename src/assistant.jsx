import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chip, ChipGroup, SortableTable } from "./components.jsx";
import { fmt } from "./charts.jsx";

/**
 * Dewey: the assistant, as a floating launcher and a docked panel.
 *
 * One implementation, reachable from every page - a second full-page copy of a chat would be two
 * things to keep in step. The character exists for a practical reason as well as a friendly one: an
 * answer that comes from a named figure reads as *someone's* answer, which is honest here, because
 * every figure is a query result and not the dashboard speaking ex cathedra.
 */
const DEWEY = {
  name: "Dewey",
  // named after the decimal classification: the one who knows where everything is filed
  role: "knows where every paper is filed",
  greeting: "Ask me anything about the papers, people and venues in dblp — who publishes the most, "
          + "how a topic grew, where a paper appeared, or what has been written on a subject. "
          + "I look up the real records for every answer.",
  working: "looking that up…",
};

/* ------------------------------------------------------------------ the character ---------- */
/**
 * Drawn rather than drawn-on: inline SVG so it inherits the theme, needs no asset, and can change
 * expression. `mood` is "idle" | "thinking" | "talking"; the antenna pulses while he works.
 */
export function Dewey({ size = 44, mood = "idle", title = DEWEY.name }) {
  const thinking = mood === "thinking";
  return (
    <svg className={"dewey dewey-" + mood} width={size} height={size} viewBox="0 0 64 64"
         role="img" aria-label={title}>
      {/* Shapes are kept few and bold, because this is read at 40px on the launcher as often as at
          104px on the introduction: an arc that reads as a lab coat at full size is a smudge small. */}
      {/* antenna: the only thing that says "not a person" */}
      <line x1="32" y1="10" x2="32" y2="6" stroke="var(--ink-2)" strokeWidth="2" strokeLinecap="round" />
      <circle className="dewey-bulb" cx="32" cy="4.5" r="2.8" fill="var(--s3)" />
      {/* the shoulders, drawn first so the head sits on them */}
      <path d="M8 62c0-10.5 10-16 24-16s24 5.5 24 16z" fill="var(--surface-2)"
            stroke="var(--ink-2)" strokeWidth="2" strokeLinejoin="round" />
      <path d="M25 47.5 32 55l7-7.5" fill="none" stroke="var(--ink-2)" strokeWidth="2" strokeLinejoin="round" />
      {/* a pocket with a pen: a researcher, not an appliance */}
      <rect x="41" y="52" width="8" height="7" rx="1.5" fill="none" stroke="var(--ink-2)" strokeWidth="1.6" />
      <line x1="45" y1="50" x2="45" y2="56" stroke="var(--s2)" strokeWidth="2.4" strokeLinecap="round" />
      {/* head, with the earpieces that give the silhouette its width */}
      <rect x="11" y="22" width="6" height="11" rx="3" fill="var(--surface-2)" stroke="var(--ink-2)" strokeWidth="2" />
      <rect x="47" y="22" width="6" height="11" rx="3" fill="var(--surface-2)" stroke="var(--ink-2)" strokeWidth="2" />
      <rect x="15" y="10" width="34" height="33" rx="12" fill="var(--surface-2)"
            stroke="var(--ink-2)" strokeWidth="2" />
      {/* glasses */}
      <g stroke="var(--accent)" strokeWidth="2" fill="none">
        <circle cx="24.5" cy="25" r="6" />
        <circle cx="39.5" cy="25" r="6" />
        <path d="M30.5 25h3" strokeLinecap="round" />
      </g>
      {/* eyes: dots at rest, a level gaze while thinking */}
      {thinking ? (
        <g stroke="var(--ink)" strokeWidth="2.4" strokeLinecap="round">
          <path d="M22 25.5h5" />
          <path d="M37 25.5h5" />
        </g>
      ) : (
        <g fill="var(--ink)">
          <circle cx="24.5" cy="25.5" r="2.2" />
          <circle cx="39.5" cy="25.5" r="2.2" />
        </g>
      )}
      {/* mouth */}
      {mood === "talking"
        ? <path d="M27.5 34.5q4.5 5 9 0" fill="none" stroke="var(--ink-2)" strokeWidth="2" strokeLinecap="round" />
        : <path d="M28 35q4 3 8 0" fill="none" stroke="var(--ink-2)" strokeWidth="2" strokeLinecap="round" />}
    </svg>
  );
}

/** The launcher: fixed, round, his face on it. Hidden while the panel is open. */
export function DeweyButton({ onClick, hidden, busy }) {
  return (
    <button type="button" className={"deweyfab" + (hidden ? " gone" : "")} onClick={onClick}
            aria-label={`Ask ${DEWEY.name}`} title={`Ask ${DEWEY.name} a question`}>
      <Dewey size={40} mood={busy ? "thinking" : "idle"} title="" />
      <span className="deweyfablabel">Ask {DEWEY.name}</span>
    </button>
  );
}

/* ------------------------------------------------------------------ the service ------------- */
const TOKEN_KEY = "dblp-chat-token";
const readToken = () => { try { return localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; } };
const writeToken = (t) => { try { localStorage.setItem(TOKEN_KEY, t); } catch { /* private mode */ } };

function useChatStatus(token, bump, enabled) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  useEffect(() => {
    if (!enabled) return undefined;
    let dead = false;
    (async () => {
      try {
        const res = await fetch("/api/chat/status", { headers: token ? { "X-Chat-Token": token } : {} });
        if (res.status === 401) throw new Error("needs-token");
        const body = await res.json();
        if (!dead) setState({ data: body, error: null, loading: false });
      } catch (e) {
        if (!dead) setState({ data: null, error: e.message, loading: false });
      }
    })();
    return () => { dead = true; };
  }, [token, bump, enabled]);
  return state;
}

async function streamAnswer({ question, history, token, onEvent, signal }) {
  const res = await fetch("/api/chat/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(token ? { "X-Chat-Token": token } : {}) },
    body: JSON.stringify({ question, history }),
    signal,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    // a validation failure arrives as a list of objects; handed to Error as-is it reads "[object Object]"
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
    throw new Error(res.status === 401 ? "needs-token" : (detail || `The service answered ${res.status}`));
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";
    for (const chunk of chunks) {
      const line = chunk.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      try { onEvent(JSON.parse(line.slice(5))); } catch { /* a partial frame */ }
    }
  }
}


/* ------------------------------------------------------------------ chats -------------------- */
const CHATS_KEY = "dblp-dewey-chats";
const MAX_CHATS = 20;
const MAX_TURNS = 20;

/** A turn as it is kept between visits: the tables are dropped (they can be large and the chips
 *  still read without them), everything that explains the answer is kept. */
function slim(turn) {
  return {
    question: turn.question,
    answer: turn.answer,
    error: turn.error || null,
    done: turn.done ? { seconds: turn.done.seconds, tools: turn.done.tools, cached: turn.done.cached,
                        cost_usd: turn.done.cost_usd, model: turn.done.model, usage: turn.done.usage,
                        memory: turn.done.memory } : null,
    tools: (turn.tools || []).map((t) => ({ type: "tool", name: t.name, arguments: t.arguments,
                                            ms: t.ms, summary: t.summary, note: t.note, link: t.link,
                                            refused: t.refused, sql: t.sql })),
  };
}

const title = (q) => (q.length > 48 ? q.slice(0, 47).trimEnd() + "…" : q);

/**
 * Conversations live in this browser, not on the server: the question log deliberately records
 * nothing that identifies anyone, and a chat history is exactly that. One list of chats, one active.
 */
function useChats() {
  const [chats, setChats] = useState(() => {
    try {
      const raw = JSON.parse(localStorage.getItem(CHATS_KEY) || "[]");
      return Array.isArray(raw) ? raw : [];
    } catch { return []; }
  });
  const [activeId, setActiveId] = useState(null);

  const persist = useCallback((next) => {
    setChats(next);
    try { localStorage.setItem(CHATS_KEY, JSON.stringify(next.slice(0, MAX_CHATS))); }
    catch { /* private mode, or the quota: the conversation still works, it just will not come back */ }
  }, []);

  const active = chats.find((c) => c.id === activeId) || null;

  const save = useCallback((id, turns) => {
    setChats((all) => {
      const first = turns[0];
      const existing = all.find((c) => c.id === id);
      const chat = {
        id,
        title: existing?.title || (first ? title(first.question) : "New chat"),
        at: Date.now(),
        turns: turns.slice(-MAX_TURNS).map(slim),
      };
      const next = [chat, ...all.filter((c) => c.id !== id)].slice(0, MAX_CHATS);
      try { localStorage.setItem(CHATS_KEY, JSON.stringify(next)); } catch { /* ignore */ }
      return next;
    });
  }, []);

  const remove = useCallback((id) => {
    persist(chats.filter((c) => c.id !== id));
    setActiveId((current) => (current === id ? null : current));
  }, [chats, persist]);

  return { chats, active, activeId, setActiveId, save, remove, clear: () => persist([]) };
}

/* ------------------------------------------------------------------ pieces ------------------ */
/** What each lookup did, in words. The raw call stays inside the chip for anyone who wants it. */
const TOOL_LABEL = {
  dataset_facts: "checked how big dblp is", docs_lookup: "checked the definitions",
  model_cards: "checked the measured accuracy", resolve_author: "found the author",
  author_profile: "read the author’s record", author_papers: "listed their papers",
  namesakes: "counted people with that name", coauthors: "listed co-authors",
  pair_papers: "checked papers they wrote together", authors_in_both: "compared two venues’ authors",
  resolve_venue: "found the venue", venue_profile: "read the venue’s record",
  top_venues: "ranked venues", top_authors: "ranked authors",
  most_shared_names: "ranked the most shared names", count_papers: "counted papers",
  papers_timeseries: "traced it year by year", title_terms: "traced the words in titles",
  rising_words: "compared title words between two periods", search_papers: "searched 5.4M titles",
  paper_detail: "opened the record", predict_venue: "asked the venue model",
  predict_coauthors: "asked the collaboration model", run_sql: "ran a one-off query",
  network_shape: "measured the whole network", central_authors: "ranked authors by centrality",
  author_centrality: "placed them in the network", search_abstracts: "read the abstracts of the closest papers",
};
const toolLabel = (name) => TOOL_LABEL[name] || name.replace(/_/g, " ");

function pick(list, n) {
  const copy = list.slice();
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, n);
}

/**
 * Suggested questions: two easy ones as chips, then three of the hard cases as small cards with a line
 * saying what makes each one hard. Every suggestion is a question the gold set verifies, so none of
 * them is a demo that fails. "Show others" deals a new three.
 */
function Suggestions({ examples, ask }) {
  // older servers sent plain strings; treat those as basic questions rather than breaking
  const all = (examples || []).map((e) => (typeof e === "string" ? { q: e, level: "basic" } : e));
  const [deal, setDeal] = useState(0);
  const shown = useMemo(() => ({
    basics: pick(all.filter((e) => e.level !== "hard"), 2),
    hard: pick(all.filter((e) => e.level === "hard"), 3),
  // keyed on the content, not the array: a status refresh must not reshuffle under the reader
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [deal, all.map((e) => e.q).join("|")]);
  const hardTotal = all.filter((e) => e.level === "hard").length;
  if (!all.length) return null;
  return (
    <div className="suggest">
      <ChipGroup>
        {shown.basics.map((e) => <Chip key={e.q} label={e.q} on={false} onClick={() => ask(e.q)} />)}
      </ChipGroup>
      {shown.hard.length ? (
        <>
          <div className="suggesthead">
            <span>Or try a hard one</span>
            {hardTotal > shown.hard.length ? (
              <button type="button" className="linkbtn" onClick={() => setDeal((n) => n + 1)}>
                show others
              </button>
            ) : null}
          </div>
          <ul className="hardlist">
            {shown.hard.map((e) => (
              <li key={e.q}>
                <button type="button" className="hardq" onClick={() => ask(e.q)}>
                  <span className="hardtext">{e.q}</span>
                  {e.why ? <span className="hardwhy">{e.why}</span> : null}
                  {e.then ? <span className="hardthen">then ask: “{e.then}”</span> : null}
                </button>
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}

/** The papers an answer cites as [1]-[5]: title, venue and year, a DOI link, the abstract on request. */
function Sources({ rows }) {
  return (
    <ol className="sources">
      {rows.map((r) => (
        <li key={r.key}>
          <span className="srcn num">[{r.n}]</span>
          <span className="srcbody">
            <span className="srctitle">{r.title}</span>
            <span className="srcmeta">{[r.venue, r.year].filter(Boolean).join(", ")}</span>
            {r.doi ? (
              <a className="srcdoi" href={`https://doi.org/${r.doi}`} target="_blank" rel="noreferrer">DOI</a>
            ) : null}
            {r.abstract && r.abstract !== "(no abstract available)" ? (
              <details className="srcabs"><summary>abstract</summary><p>{r.abstract}</p></details>
            ) : <span className="srcmeta">no abstract</span>}
          </span>
        </li>
      ))}
    </ol>
  );
}

/** One tool call, collapsed to a single quiet line until asked. */
function ToolCall({ event, go }) {
  const [open, setOpen] = useState(false);
  const rows = event.rows || [];
  const columns = (event.columns || []).slice(0, 5);
  return (
    <div className={"toolcall" + (event.refused ? " refused" : "")}>
      <button type="button" className="toolhead" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="toolchevron" aria-hidden="true">{open ? "▾" : "▸"}</span>
        <span className="toolname">{toolLabel(event.name)}</span>
        <span className="toolms num">{event.ms} ms</span>
      </button>
      {open ? (
        <div className="toolbody">
          <div className="toolsummary">{event.summary}</div>
          <div className="toolargs mono">{event.name}({JSON.stringify(event.arguments || {})})</div>
          {event.sql ? <pre className="recordxml">{event.sql}</pre> : null}
          {event.name === "search_abstracts" && rows.length ? <Sources rows={rows} /> : null}
          {event.name !== "search_abstracts" && rows.length && columns.length ? (
            <SortableTable rows={rows.slice(0, 8)}
                           columns={columns.map((c) => ({
                             key: c, label: c.replace(/_/g, " "),
                             num: typeof rows[0][c] === "number",
                             render: (r) => (typeof r[c] === "number" ? fmt.comma(r[c]) : String(r[c] ?? "—")),
                           }))} />
          ) : null}
          {event.note ? <div className="toolnote">{event.note}</div> : null}
          {event.link?.page ? (
            <button type="button" className="btn" onClick={() => go(event.link.page, event.link)}>
              Open the {event.link.page} page
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function Turn({ turn, go }) {
  const busy = !turn.answer && !turn.error;
  return (
    <div className="turn">
      <div className="asked">{turn.question}</div>
      <div className="said">
        <Dewey size={30} mood={busy ? "thinking" : "talking"} title="" />
        <div className="saidbody">
          {turn.tools.length ? (
            <div className="toolcalls">{turn.tools.map((t, i) => <ToolCall key={i} event={t} go={go} />)}</div>
          ) : null}
          {busy ? <div className="thinkingline">{turn.status || "thinking"}<span className="dots" /></div> : null}
          {turn.answer ? <div className="answer">{turn.answer}</div> : null}
          {turn.answer && /\[\d\]/.test(turn.answer) ? (() => {
            // an answer that cites [n] shows what it cites, under it, without opening the lookup
            const found = [...turn.tools].reverse().find((t) => t.name === "search_abstracts" && t.rows?.length);
            return found ? <Sources rows={found.rows} /> : null;
          })() : null}
          {turn.error ? (
            <div className="cardmsg error" role="alert">
              {turn.error}
              {/[Ll]imit|budget/.test(turn.error)
                ? " Earlier answers are still here, and anything asked before is served from cache for free."
                : null}
            </div>
          ) : null}
          {turn.done ? (
            <div className="answermeta" title={`${turn.done.usage?.input_tokens || 0} in / `
                  + `${turn.done.usage?.output_tokens || 0} out tokens · ${turn.done.model}`}>
              {turn.done.cached ? <span className="flag">cached</span> : null}
              <span>{turn.done.seconds}s</span>
              <span>{turn.done.tools.length} quer{turn.done.tools.length === 1 ? "y" : "ies"}</span>
              <span>${(turn.done.cost_usd || 0).toFixed(4)}</span>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

/** The lock screen: a proper introduction rather than a lonely password box. */
function Unlock({ onUnlock }) {
  const [draft, setDraft] = useState("");
  return (
    <div className="deweyintro">
      <Dewey size={104} />
      <h3>{DEWEY.name}</h3>
      <p className="introrole">{DEWEY.role}</p>
      <p className="intropitch">{DEWEY.greeting}</p>
      <form className="inlineform" onSubmit={(e) => { e.preventDefault(); onUnlock(draft.trim()); }}>
        <input className="textinput" type="password" value={draft} placeholder="access token" autoFocus
               aria-label="Access token" onChange={(e) => setDraft(e.target.value)} />
        <button type="submit" className="btn" disabled={!draft.trim()}>Unlock</button>
      </form>
      <p className="introfoot">{DEWEY.name} uses a paid AI model for every question, so he is behind an
        access token. It stays in this browser and nowhere else.</p>
    </div>
  );
}

/** What he can and cannot do, plus the measured numbers - out of the way until asked for. */
function About({ status }) {
  const d = status;
  if (!d) return null;
  const ev = d.evaluation;
  return (
    <details className="deweyabout">
      <summary>About {DEWEY.name}{ev ? ` · ${Math.round(100 * ev.tool_choice_accuracy)}% right query, median ${ev.median_seconds}s` : ""}</summary>
      <div className="aboutbody">
        <p><b>How he answers.</b> Every answer comes from a fresh look at dblp itself: counts,
          rankings and trends are worked out from the records, “papers about …” searches 5.4 million
          titles by meaning as well as words, and predictions come from the models trained on this
          data. Questions about what papers say are answered from the abstracts of the closest papers,
          taken from OpenAlex and cited as [1] to [5]. He is instructed to answer only from what these
          lookups return, and you can open every lookup above an answer to see exactly what it found.</p>
        {ev ? (
          <p><b>Measured on {ev.cases} questions</b> covering every kind he claims to answer, each naming
            the lookups it must use: {Math.round(100 * ev.tool_choice_accuracy)}% reached for the right
            ones, {Math.round(100 * ev.refusal_accuracy)}% of out-of-scope questions were refused,
            {" "}{Math.round(100 * ev.grounded_share)}% of answers came after a query, median
            {" "}{ev.median_seconds}s, ${ev.total_cost_usd} for the run
            {ev.run_at ? ` (${ev.run_at.slice(0, 10)})` : ""}.</p>
        ) : null}
        <p><b>What he cannot do.</b> {d.cannot_answer}</p>
        <p className="muted">{d.tools?.length} queries available · dump {d.dump?.latest_mdate} · last
          complete year {d.dump?.last_full_year}
          {d.budget ? ` · $${d.budget.usd_left?.toFixed(2)} of today’s budget left` : ""}</p>
      </div>
    </details>
  );
}

/* ------------------------------------------------------------------ the panel --------------- */
export function DeweyPanel({ open, onClose, go, onBusy }) {
  const [token, setToken] = useState(readToken);
  const [bump, setBump] = useState(0);
  const status = useChatStatus(token, bump, open);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState([]);
  const [busy, setBusy] = useState(false);
  const { chats, active, activeId, setActiveId, save, remove } = useChats();
  const [listOpen, setListOpen] = useState(false);
  const chatsRef = useRef(chats);
  const abort = useRef(null);
  const bottom = useRef(null);
  const input = useRef(null);

  useEffect(() => { onBusy?.(busy); }, [busy, onBusy]);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [turns]);
  useEffect(() => { if (open) input.current?.focus(); }, [open]);
  useEffect(() => { chatsRef.current = chats; }, [chats]);
  // What the panel shows changes only when the reader switches chats - never as a side effect of the
  // id changing. It used to be an effect keyed on the id, and the first question of a new chat mints
  // an id: the effect then "loaded" that chat - not saved yet, so empty - over the question being
  // answered, which vanished. The second try worked only because the id already existed.
  const openChat = useCallback((id) => {
    const saved = id ? chatsRef.current.find((c) => c.id === id) : null;
    setActiveId(id);
    setTurns(saved ? saved.turns.map((t) => ({ ...t, tools: t.tools || [] })) : []);
    setListOpen(false);
  }, [setActiveId]);
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  useEffect(() => () => abort.current?.abort(), []);

  const ask = useCallback(async (text) => {
    const q = (text ?? question).trim();
    if (!q || busy) return;
    setQuestion("");
    setBusy(true);
    setListOpen(false);
    const chatId = activeId || `c${Date.now().toString(36)}`;
    if (!activeId) setActiveId(chatId);
    const index = turns.length;
    // the server's memory of each answer - the answer plus the pages behind it - so a follow-up like
    // "the second one" can be resolved; older saved chats only have the answer
    const history = turns.flatMap((t) => (t.answer
      ? [{ role: "user", content: t.question },
         { role: "assistant", content: t.done?.memory || t.answer }] : []));
    setTurns((all) => [...all, { question: q, tools: [], answer: "", status: "thinking", done: null, error: null }]);
    const patch = (fn) => setTurns((all) => {
      const next = all.map((t, i) => (i === index ? fn(t) : t));
      if (next[index]?.done || next[index]?.error) save(chatId, next);   // keep it once it is settled
      return next;
    });
    const ctrl = new AbortController();
    abort.current = ctrl;
    try {
      await streamAnswer({
        question: q, history, token, signal: ctrl.signal,
        onEvent: (e) => {
          if (e.type === "status") patch((t) => ({ ...t, status: e.text }));
          else if (e.type === "tool") patch((t) => ({ ...t, tools: [...t.tools, e] }));
          else if (e.type === "token") patch((t) => ({ ...t, answer: t.answer + e.text, status: null }));
          else if (e.type === "done") patch((t) => ({ ...t, done: e, status: null, answer: t.answer || e.answer }));
          else if (e.type === "error") patch((t) => ({ ...t, error: String(e.message || "Something went wrong."), status: null }));
        },
      });
    } catch (e) {
      if (e.name !== "AbortError") {
        const needs = e.message === "needs-token";
        patch((t) => ({ ...t, error: needs ? "That token was not accepted." : e.message, status: null }));
        if (needs) setToken("");
      }
    } finally {
      setBusy(false);
      abort.current = null;
    }
  }, [question, busy, turns, token, activeId, setActiveId, save]);

  const d = status.data;
  const needsToken = status.error === "needs-token";

  return (
    <>
      <div className={"deweyscrim" + (open ? " show" : "")} onClick={onClose} aria-hidden="true" />
      <aside className={"deweypanel" + (open ? " open" : "")} role="dialog" aria-modal="true"
             aria-label={`Ask ${DEWEY.name}`}>
        <header className="deweyhead">
          <Dewey size={38} mood={busy ? "thinking" : "idle"} title="" />
          <div className="deweywho">
            <b>{DEWEY.name}</b>
            <span>{busy ? DEWEY.working : (active ? active.title : DEWEY.role)}</span>
          </div>
          {turns.length ? (
            <button type="button" className="deweyicon" title="New chat" aria-label="New chat"
                    onClick={() => openChat(null)}>＋</button>
          ) : null}
          {chats.length ? (
            <button type="button" className={"deweyicon" + (listOpen ? " on" : "")} title="Earlier chats"
                    aria-label="Earlier chats" aria-expanded={listOpen}
                    onClick={() => setListOpen(!listOpen)}>☰</button>
          ) : null}
          <button type="button" className="deweyicon" onClick={onClose} title="Close"
                  aria-label="Close">✕</button>
        </header>

        <div className="deweybody">
          {listOpen && chats.length ? (
            <ul className="chatlist">
              {chats.map((c) => (
                <li key={c.id} className={c.id === activeId ? "on" : undefined}>
                  <button type="button" className="chatopen"
                          onClick={() => openChat(c.id)}>
                    <span className="chattitle">{c.title}</span>
                    <span className="chatwhen">{c.turns.length} question{c.turns.length === 1 ? "" : "s"}
                      {" · "}{new Date(c.at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</span>
                  </button>
                  <button type="button" className="deweyicon" title="Delete this chat"
                          aria-label={`Delete ${c.title}`}
                          onClick={() => { remove(c.id); if (c.id === activeId) setTurns([]); }}>✕</button>
                </li>
              ))}
            </ul>
          ) : null}
          {needsToken ? (
            <Unlock onUnlock={(t) => { writeToken(t); setToken(t); setBump((b) => b + 1); }} />
          ) : (
            <>
              {d && !d.configured ? (
                <div className="cardmsg">No model provider is configured for the assistant
                  (<code>OPENAI_API_KEY</code>). Everything else on this site works without it.</div>
              ) : null}
              {status.error && !needsToken ? (
                <div className="cardmsg error">Couldn’t reach {DEWEY.name}: {status.error}</div>
              ) : null}

              {turns.length === 0 ? (
                <div className="deweyintro compact">
                  <Dewey size={76} />
                  <h3>{DEWEY.name}</h3>
                  <p className="introrole">{DEWEY.role}</p>
                  <p className="intropitch">{DEWEY.greeting}</p>
                  <Suggestions examples={d?.examples} ask={ask} />
                  <p className="introfoot">
                    <a href="#research" onClick={(e) => { e.preventDefault(); onClose(); go("research"); }}>
                      How much does retrieval help an assistant like {DEWEY.name}? Read the study →
                    </a>
                  </p>
                </div>
              ) : (
                <div className="thread">
                  {turns.map((t, i) => <Turn key={i} turn={t} go={go} />)}
                  <div ref={bottom} />
                </div>
              )}
            </>
          )}
        </div>

        {!needsToken ? (
          <footer className="deweyfoot">
            <form onSubmit={(e) => { e.preventDefault(); ask(); }}>
              <input ref={input} className="askinput" value={question}
                     maxLength={d?.limits?.question_chars || 400}
                     placeholder={`Ask ${DEWEY.name} anything about dblp…`} aria-label="Your question"
                     onChange={(e) => setQuestion(e.target.value)} />
              {busy
                ? <button type="button" className="btn" onClick={() => abort.current?.abort()}>Stop</button>
                : <button type="submit" className="btn primary" disabled={!question.trim()}>Ask</button>}
            </form>
            <About status={d} />
          </footer>
        ) : null}
      </aside>
    </>
  );
}

export const ASSISTANT_NAME = DEWEY.name;
