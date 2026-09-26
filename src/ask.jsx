import { useCallback, useEffect, useRef, useState } from "react";
import { PageHead, Card, Callout, EmptyNote, SortableTable, Chip, ChipGroup, Spinner } from "./components.jsx";
import { fmt } from "./charts.jsx";

const TOKEN_KEY = "dblp-chat-token";
const readToken = () => { try { return localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; } };
const writeToken = (t) => { try { localStorage.setItem(TOKEN_KEY, t); } catch { /* private mode */ } };

/** The service's own description of itself: models, freshness, budget left, example questions. */
function useChatStatus(token, bump) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  useEffect(() => {
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
  }, [token, bump]);
  return state;
}

/** POST the question and read the server-sent event stream, calling `onEvent` as things happen. */
async function streamAnswer({ question, history, token, onEvent, signal }) {
  const res = await fetch("/api/chat/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(token ? { "X-Chat-Token": token } : {}) },
    body: JSON.stringify({ question, history }),
    signal,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(res.status === 401 ? "needs-token" : (body.detail || `The service answered ${res.status}`));
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
      try { onEvent(JSON.parse(line.slice(5))); } catch { /* a partial frame: ignore */ }
    }
  }
}

function ToolCall({ event, go }) {
  const [open, setOpen] = useState(false);
  const rows = event.rows || [];
  const columns = (event.columns || []).slice(0, 6);
  return (
    <div className={"toolcall" + (event.refused ? " refused" : "")}>
      <button type="button" className="toolhead" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="toolname mono">{event.name}</span>
        <span className="toolargs mono">{JSON.stringify(event.arguments || {}).slice(1, -1) || "—"}</span>
        <span className="toolms num">{event.ms} ms</span>
        <span className="toolchevron" aria-hidden="true">{open ? "▾" : "▸"}</span>
      </button>
      {open ? (
        <div className="toolbody">
          <div className="toolsummary">{event.summary}</div>
          {event.sql ? <pre className="recordxml">{event.sql}</pre> : null}
          {rows.length && columns.length ? (
            <SortableTable rows={rows.slice(0, 12)}
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

function Exchange({ turn, go }) {
  return (
    <div className="exchange">
      <div className="asked">{turn.question}</div>
      {turn.tools.length ? (
        <div className="toolcalls">
          {turn.tools.map((t, i) => <ToolCall key={i} event={t} go={go} />)}
        </div>
      ) : null}
      {turn.status && !turn.answer ? <div className="cardmsg"><Spinner /> {turn.status}…</div> : null}
      {turn.answer ? <div className="answer">{turn.answer}</div> : null}
      {turn.error ? <div className="cardmsg error" role="alert">{turn.error}</div> : null}
      {turn.done ? (
        <div className="answermeta">
          {turn.done.cached ? <span className="flag">from cache</span> : null}
          <span>{turn.done.seconds}s</span>
          <span>{turn.done.rounds} round{turn.done.rounds === 1 ? "" : "s"}</span>
          <span>{turn.done.tools.length} tool call{turn.done.tools.length === 1 ? "" : "s"}</span>
          <span>{(turn.done.usage?.input_tokens || 0) + (turn.done.usage?.output_tokens || 0)} tokens</span>
          <span>${(turn.done.cost_usd || 0).toFixed(4)}</span>
          <span className="muted">{turn.done.model}</span>
        </div>
      ) : null}
    </div>
  );
}

export function PageAsk({ go }) {
  const [token, setToken] = useState(readToken);
  const [tokenDraft, setTokenDraft] = useState("");
  const [bump, setBump] = useState(0);
  const status = useChatStatus(token, bump);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState([]);
  const [busy, setBusy] = useState(false);
  const abort = useRef(null);
  const bottom = useRef(null);

  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [turns]);
  useEffect(() => () => abort.current?.abort(), []);

  const ask = useCallback(async (text) => {
    const q = (text ?? question).trim();
    if (!q || busy) return;
    setQuestion("");
    setBusy(true);
    const index = turns.length;
    const history = turns.flatMap((t) => (t.answer
      ? [{ role: "user", content: t.question }, { role: "assistant", content: t.answer }] : []));
    setTurns((all) => [...all, { question: q, tools: [], answer: "", status: "thinking", done: null, error: null }]);
    const patch = (fn) => setTurns((all) => all.map((t, i) => (i === index ? fn(t) : t)));
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
          else if (e.type === "error") patch((t) => ({ ...t, error: e.message, status: null }));
        },
      });
    } catch (e) {
      if (e.name !== "AbortError") {
        patch((t) => ({ ...t, error: e.message === "needs-token" ? "This assistant needs an access token." : e.message,
                        status: null }));
        if (e.message === "needs-token") setToken("");
      }
    } finally {
      setBusy(false);
      abort.current = null;
    }
  }, [question, busy, turns, token]);

  const d = status.data;
  const needsToken = status.error === "needs-token";

  if (needsToken) {
    return (
      <>
        <PageHead eyebrow="Ask · natural language" title="Ask dblp a question">
          This assistant calls a language model for every question, so it is behind a token.
        </PageHead>
        <Card title="Access token" sub="Paste the token you were given. It is kept in this browser only.">
          <form className="inlineform" onSubmit={(e) => { e.preventDefault(); writeToken(tokenDraft.trim()); setToken(tokenDraft.trim()); setBump((b) => b + 1); }}>
            <input className="textinput" type="password" value={tokenDraft} placeholder="access token"
                   aria-label="Access token" onChange={(e) => setTokenDraft(e.target.value)} />
            <button type="submit" className="btn" disabled={!tokenDraft.trim()}>Unlock</button>
          </form>
        </Card>
      </>
    );
  }

  return (
    <>
      <PageHead eyebrow="Ask · natural language" title="Ask dblp a question">
        Questions are answered by querying the dump, not by guessing: every answer shows the tools it
        called, their results and their timings. Counts, rankings and trends are SQL; “papers about …”
        is the hybrid search; predictions come from the three models.
      </PageHead>

      {d && !d.configured ? (
        <Callout>The assistant has no model provider configured yet: set <code>OPENAI_API_KEY</code> for
          the <code>chatapi</code> service. Everything else on this site works without it.</Callout>
      ) : null}
      {d && d.configured && !d.ready ? (
        <Callout>The assistant is still attaching the dump{d.error ? `: ${d.error}` : "…"}</Callout>
      ) : null}
      {status.error && status.error !== "needs-token" ? (
        <Callout>Couldn’t reach the assistant: {status.error}</Callout>
      ) : null}

      <div className="askbox">
        <form onSubmit={(e) => { e.preventDefault(); ask(); }}>
          <input className="askinput" value={question} autoFocus
                 maxLength={d?.limits?.question_chars || 400}
                 placeholder="e.g. which author has the most papers, and in which venues?"
                 aria-label="Your question" onChange={(e) => setQuestion(e.target.value)} />
          <button type="submit" className="btn primary" disabled={busy || !question.trim()}>
            {busy ? "Working…" : "Ask"}
          </button>
          {busy ? <button type="button" className="btn" onClick={() => abort.current?.abort()}>Stop</button> : null}
        </form>
        {turns.length === 0 && d?.examples ? (
          <ChipGroup>
            {d.examples.slice(0, 8).map((ex) => (
              <Chip key={ex} label={ex} on={false} onClick={() => ask(ex)} />
            ))}
          </ChipGroup>
        ) : null}
      </div>

      {turns.length ? (
        <div className="thread">
          {turns.map((t, i) => <Exchange key={i} turn={t} go={go} />)}
          <div ref={bottom} />
        </div>
      ) : null}

      <div className="grid">
        <Card title="What it can do" sub="Each question is routed to typed queries over the dump; nothing is answered from the model's own memory.">
          {() => (
            <ul className="authorlist">
              <li>Counts, rankings and trends — “how many”, “who has most”, “has X changed”</li>
              <li>One author, one venue, one paper — resolved through the author-page registry</li>
              <li>Topic search over 5.4M titles, ranked by meaning as well as words</li>
              <li>Two-step questions — “co-authors of X who also publish at Y”</li>
              <li>The three models — a bin split, next co-authors, where to publish</li>
              <li>Definitions and counting rules — “does this include preprints?”</li>
            </ul>
          )}
        </Card>
        <Card title="What it cannot do" sub="Not in dblp, so not answerable here.">
          {() => (
            <>
              <div className="cardsub">{d?.cannot_answer}</div>
              {d ? (
                <div className="answermeta">
                  <span>{d.tools?.length} tools</span>
                  <span>dump {d.dump?.latest_mdate}</span>
                  <span>last complete year {d.dump?.last_full_year}</span>
                  {d.budget ? <span>${d.budget.usd_left?.toFixed(2)} of today’s budget left</span> : null}
                  <span className="muted">{d.models?.router} → {d.models?.answers}</span>
                </div>
              ) : null}
            </>
          )}
        </Card>
      </div>
      {d?.examples?.length > 8 ? (
        <Card title="More things to try" sub="Click one to run it.">
          {() => (
            <ChipGroup>
              {d.examples.slice(8).map((ex) => <Chip key={ex} label={ex} on={false} onClick={() => ask(ex)} />)}
            </ChipGroup>
          )}
        </Card>
      ) : null}
      {!turns.length ? <EmptyNote>Nothing asked yet.</EmptyNote> : null}
    </>
  );
}
