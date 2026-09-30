import { useEffect, useRef, useState } from "react";

/**
 * Fetch a static JSON file served by nginx rather than an /api endpoint - the published dataset's
 * manifest. A 404 there means "nothing published yet", which is a state to show, not an error.
 */
export function useStatic(url, as = "json") {
  const [state, setState] = useState({ data: null, error: null, loading: true, missing: false });
  useEffect(() => {
    let cancelled = false;
    const ctrl = new AbortController();
    fetch(url, { signal: ctrl.signal, cache: "no-cache" })
      .then(async (res) => {
        if (cancelled) return;
        if (res.status === 404) {
          setState({ data: null, error: null, loading: false, missing: true });
          return;
        }
        if (!res.ok) throw new Error(`The server answered ${res.status}`);
        const body = as === "text" ? await res.text() : await res.json();
        if (!cancelled) setState({ data: body, error: null, loading: false, missing: false });
      })
      .catch((e) => {
        if (!cancelled && e.name !== "AbortError") {
          setState({ data: null, error: e.message || "Request failed", loading: false, missing: false });
        }
      });
    return () => { cancelled = true; ctrl.abort(); };
  }, [url]);
  return state;
}

/** Build /api/<path>?<params>, dropping empty values. */
export function apiUrl(path, params) {
  const qs = new URLSearchParams(
    Object.entries(params || {}).filter(([, v]) => v !== undefined && v !== null && v !== ""),
  ).toString();
  return `/api/${path}${qs ? `?${qs}` : ""}`;
}

/**
 * Fetch an API endpoint. Keeps the previous data visible while a new request (a filter change) runs.
 * While the server is still preparing its tables (HTTP 503) it retries every 5 s and exposes `warming`.
 * Pass path = null to skip fetching.
 */
export function useApi(path, params) {
  const url = path ? apiUrl(path, params) : null;
  const [state, setState] = useState({ data: null, error: null, loading: !!url, warming: null });
  const dataUrl = useRef(null);

  useEffect(() => {
    if (!url) {
      setState({ data: null, error: null, loading: false, warming: null });
      return undefined;
    }
    let cancelled = false;
    let timer;
    const ctrl = new AbortController();
    // a different resource (not just a filter change): don't show the old one while loading
    const sameResource = dataUrl.current && dataUrl.current.split("?")[0] === url.split("?")[0];

    async function run() {
      setState((s) => ({ ...s, data: sameResource ? s.data : null, loading: true, error: null }));
      try {
        const res = await fetch(url, { signal: ctrl.signal });
        const body = await res.json().catch(() => ({}));
        if (cancelled) return;
        if (res.status === 503) {
          setState((s) => ({ ...s, loading: true, warming: body.status || { message: body.detail } }));
          timer = setTimeout(run, 5000);
          return;
        }
        if (!res.ok) {
          const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
          throw new Error(detail || `The server answered ${res.status}`);
        }
        dataUrl.current = url;
        // the dashboard api wraps payloads as { data, generation }; the model server returns them bare
        const payload = body && typeof body === "object" && "data" in body ? body.data : body;
        setState({ data: payload, error: null, loading: false, warming: null });
      } catch (e) {
        if (!cancelled && e.name !== "AbortError") {
          setState((s) => ({ ...s, loading: false, error: e.message || "Request failed", warming: null }));
        }
      }
    }
    run();
    return () => {
      cancelled = true;
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [url]);

  return state;
}

/** Delay a fast-changing value (slider, text box) so we don't send a request per keystroke. */
export function useDebounced(value, ms = 350) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

/** Server status: data freshness, build progress, job output dates. Polls faster while building. */
export function useStatus() {
  const [status, setStatus] = useState(null);
  useEffect(() => {
    let timer;
    let stopped = false;
    async function poll() {
      try {
        const res = await fetch("/api/status");
        const body = await res.json();
        if (!stopped) setStatus(body);
        const busy = body?.status?.state !== "ready" || body?.status?.refreshing;
        timer = setTimeout(poll, busy ? 4000 : 60000);
      } catch {
        if (!stopped) setStatus({ status: { state: "offline", message: "The data service is not reachable" } });
        timer = setTimeout(poll, 10000);
      }
    }
    poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, []);
  return status;
}

export function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}
