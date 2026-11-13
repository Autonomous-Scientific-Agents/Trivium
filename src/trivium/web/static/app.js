"use strict";
/* trivium review UI: upload sources, follow generation, review questions, leave feedback.
   No build step, no dependencies. All model-generated text is rendered with textContent. */

const $ = (sel, root = document) => root.querySelector(sel);
const main = $("#main");
const state = { settings: null, runs: [], run: null, poll: null, filter: "all", type: "all", hideAnswers: false, files: [] };

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch (_) { return null; }
}

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* not JSON */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.status === 204 ? null : res.json();
}

const fmt = (x, d = 2) => (x === null || x === undefined ? "–" : typeof x === "number" ? x.toFixed(d) : String(x));
const pct = (x) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);
const typeLabel = { short_answer: "short answer", true_false: "true / false", multiple_choice: "multiple choice" };
const ACTIVE = new Set(["queued", "running"]);

/* ------------------------------------------------------------------ sidebar */
async function loadRuns() {
  try { state.runs = await api("/api/runs"); } catch (e) { state.runs = []; }
  renderRunList();
}

function renderRunList() {
  const list = $("#run-list");
  const current = state.run && state.run.id;
  list.replaceChildren(
    ...state.runs.map((r) =>
      h("li", {},
        h("a", { href: `#/run/${encodeURIComponent(r.id)}`, class: r.id === current ? "active" : "" },
          h("div", { class: "run-name" }, r.name),
          h("div", { class: "run-meta" },
            h("span", { class: `pill ${r.status}` }, r.status),
            r.num_questions ? h("span", {}, `${r.num_questions} q`) : null,
            r.num_up || r.num_down ? h("span", {}, `👍 ${r.num_up} · 👎 ${r.num_down}`) : null,
            r.created_at ? h("span", {}, r.created_at.replace("T", " ").slice(0, 16)) : null)))),
  );
  if (!state.runs.length) list.append(h("li", { class: "muted small" }, "No runs yet."));
}

/* ------------------------------------------------------------------ new run */
function renderNew() {
  const s = state.settings;
  state.files = [];
  const fileInput = h("input", { type: "file", multiple: true, accept: s.extensions.join(","), hidden: true });
  const fileList = h("ul", { class: "file-list" });
  const err = h("p", { class: "error", role: "alert" });

  const addFiles = (files) => {
    for (const f of files) {
      const ext = "." + (f.name.split(".").pop() || "").toLowerCase();
      if (!s.extensions.includes(ext)) { err.textContent = `Unsupported file type: ${f.name}`; continue; }
      if (f.size > s.max_upload_mb * 1024 * 1024) { err.textContent = `${f.name} is larger than ${s.max_upload_mb} MB`; continue; }
      if (!state.files.some((g) => g.name === f.name && g.size === f.size)) state.files.push(f);
    }
    drawFiles();
  };
  const drawFiles = () => {
    fileList.replaceChildren(...state.files.map((f, i) =>
      h("li", {}, h("span", {}, f.name, " ", h("span", { class: "muted small" }, `${Math.ceil(f.size / 1024)} KB`)),
        h("button", { type: "button", title: "Remove", "aria-label": `Remove ${f.name}`, onclick: () => { state.files.splice(i, 1); drawFiles(); } }, "×"))));
  };
  fileInput.addEventListener("change", () => { addFiles(fileInput.files); fileInput.value = ""; });

  const drop = h("div", { class: "dropzone", tabindex: "0", role: "button", "aria-label": "Choose source files",
    onclick: () => fileInput.click(),
    onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } },
    ondragover: (e) => { e.preventDefault(); drop.classList.add("over"); },
    ondragleave: () => drop.classList.remove("over"),
    ondrop: (e) => { e.preventDefault(); drop.classList.remove("over"); addFiles(e.dataTransfer.files); },
  }, h("strong", {}, "Drop source files here"), " or click to choose", h("br"),
  h("span", { class: "muted small" }, `${s.extensions.join(" ")} · up to ${s.max_upload_mb} MB each`));

  const types = ["short_answer", "true_false", "multiple_choice"];
  const field = (label, input, hint) => h("label", { class: "field" }, h("span", {}, label), input, hint ? h("span", { class: "hint" }, hint) : null);
  const name = h("input", { type: "text", name: "name", placeholder: "defaults to the first file name" });
  const num = h("input", { type: "number", name: "num_questions", min: "1", max: "500", placeholder: "auto" });
  const best = h("input", { type: "number", name: "best", min: "1", max: "200", placeholder: "off" });
  const provider = h("input", { type: "text", name: "provider", value: s.provider || "" });
  const model = h("input", { type: "text", name: "model", value: s.model || "" });
  const maxChunks = h("input", { type: "number", name: "max_chunks", min: "1", placeholder: "all" });
  const seed = h("input", { type: "number", name: "seed", placeholder: "config" });
  const persona = h("textarea", { name: "personas", rows: "2", placeholder: "optional, one per line, e.g. a new user who just got their first allocation and wants to run a GPU job" });
  const text = h("textarea", { name: "text", rows: "4", placeholder: "…or paste text here" });
  const submit = h("button", { class: "btn primary", type: "submit" }, "Generate questions");

  const form = h("form", { class: "panel", onsubmit: async (e) => {
    e.preventDefault();
    err.textContent = "";
    const fd = new FormData();
    for (const f of state.files) fd.append("files", f, f.name);
    fd.append("text", text.value);
    const intOrNull = (el) => (el.value.trim() === "" ? null : parseInt(el.value, 10));
    const qtypes = types.filter((t) => form.querySelector(`input[value=${t}]`).checked);
    fd.append("settings", JSON.stringify({
      name: name.value.trim(),
      mode: form.querySelector("input[name=mode]:checked").value,
      num_questions: intOrNull(num), best: intOrNull(best), question_types: qtypes,
      provider: provider.value.trim() || null, model: model.value.trim() || null,
      max_chunks: intOrNull(maxChunks), seed: intOrNull(seed),
      personas: persona.value.split("\n").map((x) => x.trim()).filter(Boolean),
    }));
    if (!state.files.length && !text.value.trim()) { err.textContent = "Add at least one file or paste some text."; return; }
    if (!qtypes.length) { err.textContent = "Pick at least one question type."; return; }
    submit.disabled = true; submit.textContent = "Uploading…";
    try {
      const r = await api("/api/runs", { method: "POST", body: fd });
      await loadRuns();
      location.hash = `#/run/${encodeURIComponent(r.id)}`;
    } catch (ex) {
      err.textContent = ex.message; submit.disabled = false; submit.textContent = "Generate questions";
      loadRuns();
    }
  } },
  h("h2", { style: "margin-top:0" }, "Sources"), drop, fileInput, fileList, h("div", { style: "margin-top:10px" }, text),
  h("h2", {}, "Settings"),
  h("div", { class: "form-grid" },
    field("Run name", name),
    h("div", { class: "field" }, h("span", {}, "Mode"), h("div", { class: "choices" },
      ...["trivia", "rag"].map((m) => h("label", {}, h("input", { type: "radio", name: "mode", value: m, checked: m === (s.mode === "rag" ? "rag" : "trivia") }), m === "rag" ? "RAG benchmark" : "Trivia")))),
    field("Questions", num, "target count; auto-sized from the sources if empty"),
    field("Best of (K)", best, "generate a larger pool and keep the K best"),
  ),
  h("div", { class: "field", style: "margin-top:14px" }, h("span", {}, "Question types"), h("div", { class: "choices" },
    ...types.map((t) => h("label", {}, h("input", { type: "checkbox", value: t, checked: s.question_types.includes(t) }), typeLabel[t])))),
  h("label", { class: "field", style: "margin-top:14px" }, h("span", {}, "Persona"), persona,
    h("span", { class: "hint" }, "who asks: steers which facts are asked about and how, and adds questions asked from the table of contents only")),
  h("details", { class: "advanced" }, h("summary", {}, "Model and limits"),
    h("div", { class: "form-grid" },
      field("Provider", provider, "openai · anthropic · argo · mock"), field("Model", model),
      field("Max chunks", maxChunks, "only use the first N chunks (quick tests)"), field("Seed", seed)),
    s.config ? h("p", { class: "muted small" }, "Other settings come from ", h("code", {}, s.config), ".") : null),
  h("div", { class: "form-actions" }, submit, err));

  main.replaceChildren(h("h1", {}, "New run"), h("p", { class: "muted" }, "Upload documents to generate trivia questions from them, then review the results."), form);
}

/* ------------------------------------------------------------------ run view */
async function openRun(id) {
  stopPoll();
  main.replaceChildren(h("p", { class: "muted" }, "Loading…"));
  try {
    state.run = await api(`/api/runs/${encodeURIComponent(id)}`);
  } catch (e) {
    main.replaceChildren(h("div", { class: "empty" }, `Could not load run: ${e.message}`));
    return;
  }
  renderRunList();
  renderRun();
  if (ACTIVE.has(state.run.status)) startPoll(id);
}

function stopPoll() { if (state.poll) clearInterval(state.poll); state.poll = null; }

function startPoll(id) {
  state.poll = setInterval(async () => {
    if (!state.run || state.run.id !== id) return stopPoll();
    let st;
    try { st = await api(`/api/runs/${encodeURIComponent(id)}/status`); } catch (_) { return; }
    if (!ACTIVE.has(st.status)) { stopPoll(); loadRuns(); openRun(id); return; }
    Object.assign(state.run, st);
    const panel = $("#status-panel");
    if (panel) panel.replaceWith(statusPanel(state.run));
  }, 2000);
}

function statusPanel(r) {
  const head = r.status === "queued"
    ? `Queued${r.queue_position ? ` (position ${r.queue_position})` : ""}…`
    : r.status === "running" ? "Generating questions…"
      : r.status === "failed" ? "Generation failed" : r.status === "interrupted" ? "Interrupted (server restarted)" : r.status;
  const retry = ["failed", "interrupted"].includes(r.status) && r.settings
    ? h("button", { class: "btn", onclick: async () => { try { await api(`/api/runs/${encodeURIComponent(r.id)}/retry`, { method: "POST" }); openRun(r.id); loadRuns(); } catch (e) { alert(e.message); } } }, "Retry")
    : null;
  const log = h("pre", { class: "log" }, (r.log || []).join("\n") || "waiting for log output…");
  const panel = h("div", { class: "panel", id: "status-panel" },
    h("div", { class: "run-head" }, h("strong", {}, head), retry),
    r.error ? h("p", { class: "error" }, r.error) : null, log);
  requestAnimationFrame(() => { log.scrollTop = log.scrollHeight; });
  return panel;
}

function renderRun() {
  const r = state.run;
  const ds = r.dataset || {};
  const models = ds.models || {};
  const gen = models.question_generator || Object.values(models)[0];
  const head = h("div", { class: "run-head" },
    h("div", {},
      h("h1", {}, r.name),
      h("div", { class: "facts" },
        h("span", { class: `pill ${r.status}` }, r.status),
        r.mode ? h("span", {}, "mode ", h("b", {}, r.mode)) : null,
        h("span", {}, "sources ", h("b", {}, (r.sources || []).join(", ") || "–")),
        gen ? h("span", {}, "model ", h("b", {}, gen)) : null,
        ds.generated_candidates ? h("span", {}, h("b", {}, ds.accepted_questions), ` accepted of ${ds.generated_candidates} candidates`) : null,
        ds.selected_top ? h("span", {}, `best ${ds.selected_top} by `, h("b", {}, ds.selected_by)) : null)),
    r.questions.length ? h("div", { class: "choices" },
      h("a", { class: "btn", href: `/api/runs/${encodeURIComponent(r.id)}/feedback.jsonl` }, "Export feedback"),
      h("a", { class: "btn", href: `/api/runs/${encodeURIComponent(r.id)}/questions.json` }, "Questions JSON")) : null);

  const parts = [head];
  if (r.status !== "done" || r.error) parts.push(statusPanel(r));
  if (r.questions.length) {
    parts.push(toolbar(), h("div", { class: "cards", id: "cards" }));
    parts.push(h("p", { class: "muted small", style: "margin-top:18px" },
      "Keys: ", h("kbd", {}, "j"), "/", h("kbd", {}, "k"), " next/previous · ", h("kbd", {}, "u"), " thumbs up · ",
      h("kbd", {}, "d"), " thumbs down · ", h("kbd", {}, "c"), " comment · ", h("kbd", {}, "a"), " show answer"));
  } else if (r.status === "done") {
    parts.push(h("div", { class: "empty" }, "This run produced no questions. See the log in the run folder for rejection reasons."));
  }
  if (r.persona_gaps && r.persona_gaps.length) {
    parts.push(h("details", { class: "panel more", style: "margin-top:18px" },
      h("summary", {}, `${r.persona_gaps.length} persona questions the sources do not fully answer`),
      h("ul", {}, ...r.persona_gaps.map((g) => h("li", {}, g.question, " ",
        h("span", { class: "muted small" }, `(${g.status}${g.style ? ", " + g.style : ""}${g.note ? ": " + g.note : ""})`))))));
  }
  main.replaceChildren(...parts);
  if (r.questions.length) renderCards();
}

function counts() {
  const r = state.run;
  let up = 0, down = 0, commentOnly = 0;
  for (const q of r.questions) {
    const f = r.feedback[q.id];
    if (!f) continue;
    if (f.rating === "up") up++; else if (f.rating === "down") down++; else if (f.comment) commentOnly++;
  }
  return { up, down, commentOnly, total: r.questions.length };
}

function toolbar() {
  const seg = (key, options) => h("div", { class: "seg", role: "group" },
    ...options.map(([v, label]) => h("button", { type: "button", "aria-pressed": String(state[key] === v),
      onclick: () => { state[key] = v; renderRun(); } }, label)));
  const types = [...new Set(state.run.questions.map((q) => q.type))];
  const c = counts();
  const w = (n) => `${(100 * n) / Math.max(1, c.total)}%`;
  return h("div", { class: "toolbar" },
    seg("filter", [["all", "All"], ["unrated", "Unrated"], ["up", "👍"], ["down", "👎"], ["commented", "Commented"]]),
    types.length > 1 ? seg("type", [["all", "All types"], ...types.map((t) => [t, typeLabel[t] || t])]) : null,
    h("label", { class: "choices small" }, h("input", { type: "checkbox", checked: state.hideAnswers,
      onchange: (e) => { state.hideAnswers = e.target.checked; store("trivium.hideAnswers", e.target.checked ? "1" : "0"); renderCards(); } }), "Quiz mode (hide answers)"),
    h("div", { class: "progress", id: "progress" },
      h("div", { class: "bar", "aria-hidden": "true" }, h("i", { class: "up", style: `width:${w(c.up)}` }), h("i", { class: "down", style: `width:${w(c.down)}` }), h("i", { class: "other", style: `width:${w(c.commentOnly)}` })),
      `${c.up + c.down} of ${c.total} rated`));
}

function visibleQuestions() {
  const r = state.run;
  return r.questions.filter((q) => {
    const f = r.feedback[q.id] || {};
    if (state.type !== "all" && q.type !== state.type) return false;
    switch (state.filter) {
      case "unrated": return !f.rating;
      case "up": return f.rating === "up";
      case "down": return f.rating === "down";
      case "commented": return !!f.comment;
      default: return true;
    }
  });
}

function renderCards() {
  const box = $("#cards");
  if (!box) return;
  const qs = visibleQuestions();
  box.replaceChildren(...(qs.length ? qs.map((q) => card(q, state.run.questions.indexOf(q) + 1)) : [h("div", { class: "empty" }, "No questions match this filter.")]));
}

function refreshProgress() {
  const p = $("#progress");
  if (p) p.replaceWith(toolbar().querySelector("#progress"));
  const s = state.runs.find((x) => x.id === state.run.id);
  if (s) { const c = counts(); s.num_up = c.up; s.num_down = c.down; renderRunList(); }
}

function answerIndex(q) {
  if (q.type !== "multiple_choice" || !q.options) return -1;
  const a = String(q.answer || "").trim().toUpperCase();
  return a.length === 1 ? a.charCodeAt(0) - 65 : q.options.indexOf(q.answer);
}

function card(q, num) {
  const r = state.run;
  const f = r.feedback[q.id] || {};
  const ev = q.evaluation || {};
  const g = q.generation || {};
  let hidden = state.hideAnswers;

  const correct = answerIndex(q);
  const picks = ev.option_pick_counts || {};
  const options = q.options ? h("ol", { class: "options" }, ...q.options.map((o, i) => {
    const letter = String.fromCharCode(65 + i);
    const n = picks[letter] ?? picks[o];
    return h("li", { class: !hidden && i === correct ? "correct" : "" },
      h("span", { class: "letter" }, letter), h("span", {}, o),
      !hidden && n ? h("span", { class: "picks", title: "closed-book picks" }, `picked ${n}×`) : null);
  })) : null;

  const answerText = q.type === "multiple_choice" && correct >= 0 && q.options
    ? `${String.fromCharCode(65 + correct)}) ${q.options[correct]}` : String(q.answer);
  const alts = (q.accepted_answers || []).filter((a) => a !== q.answer && a !== answerText && q.type === "short_answer");
  const answer = h("div", { class: "answer" },
    h("span", { class: "label" }, "Answer"),
    h("span", { class: `value${hidden ? " hidden-answer" : ""}`, title: hidden ? "click to reveal" : null,
      onclick: () => { if (hidden) { hidden = false; el.replaceWith(rebuild(false)); } } }, answerText),
    !hidden && alts.length ? h("span", { class: "alts" }, `also accepted: ${alts.join(" · ")}`) : null);

  const metrics = [
    ["quality", fmt(ev.overall_quality)], ["interestingness", fmt(ev.interestingness)],
    ["standalone", fmt(ev.standalone)], ["doc dependence", fmt(ev.document_dependence)],
    ["closed-book acc.", pct(ev.closed_book_accuracy)], ["open-book acc.", pct(ev.open_book_accuracy)],
    ["novelty", fmt(ev.novelty)], ["pairwise wins", pct(ev.pairwise_win_rate)],
  ].filter(([, v]) => v !== "–");
  const deriv = g.derivation || {};
  const derivRows = Object.entries(deriv).filter(([k, v]) => v !== null && v !== "" && !(Array.isArray(v) && !v.length) && k !== "base_answer");
  const more = h("details", { class: "more" }, h("summary", {}, "Evidence and scores"),
    ...(q.evidence || []).map((e) => h("div", { class: "evidence" },
      h("div", { class: "src" }, [e.document_id, e.section].filter(Boolean).join(" · ")), e.text)),
    metrics.length ? h("div", { class: "metrics" }, ...metrics.map(([k, v]) => h("div", { class: "metric" }, h("span", {}, k), h("b", {}, v)))) : null,
    ev.closed_book_answers && ev.closed_book_answers.length ? h("div", { class: "kv" }, h("span", {}, "closed-book answers: "), ev.closed_book_answers.join(" · ")) : null,
    ...derivRows.map(([k, v]) => h("div", { class: "kv" }, h("span", {}, `${k.replace(/_/g, " ")}: `), typeof v === "object" ? JSON.stringify(v) : String(v))),
    ev.notes ? h("div", { class: "kv" }, h("span", {}, "evaluator: "), ev.notes) : null);

  /* feedback */
  const saveState = h("span", { class: "save-state", "aria-live": "polite" });
  const comment = h("textarea", { rows: "1", placeholder: "Feedback (optional): what is wrong or good about this question?", "aria-label": `Feedback on question ${num}` });
  comment.value = f.comment || "";
  let rating = f.rating || null;
  const thumb = (v, glyph, label) => h("button", { type: "button", class: `thumb ${v}`, title: label, "aria-label": label, "aria-pressed": String(rating === v),
    onclick: () => { rating = rating === v ? null : v; save(); } }, glyph);
  const up = thumb("up", "👍", "Good question"), down = thumb("down", "👎", "Bad question");
  let timer = null, lastSent = JSON.stringify([rating, comment.value.trim()]);
  async function save() {
    clearTimeout(timer);
    const payload = JSON.stringify([rating, comment.value.trim()]);
    up.setAttribute("aria-pressed", String(rating === "up"));
    down.setAttribute("aria-pressed", String(rating === "down"));
    el.classList.toggle("rated-up", rating === "up");
    el.classList.toggle("rated-down", rating === "down");
    if (payload === lastSent) return;
    saveState.className = "save-state"; saveState.textContent = "Saving…";
    try {
      const res = await api(`/api/runs/${encodeURIComponent(r.id)}/feedback/${encodeURIComponent(q.id)}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rating, comment: comment.value, reviewer: ($("#reviewer").value || "").trim() || null }),
      });
      lastSent = payload;
      if (res.feedback) r.feedback[q.id] = res.feedback; else delete r.feedback[q.id];
      saveState.textContent = "Saved";
      refreshProgress();
    } catch (e) {
      saveState.className = "save-state err"; saveState.textContent = "Not saved";
      saveState.title = e.message;
    }
  }
  comment.addEventListener("input", () => {
    comment.style.height = "auto"; comment.style.height = `${Math.max(38, comment.scrollHeight + 2)}px`;
    clearTimeout(timer); timer = setTimeout(save, 800);
  });
  comment.addEventListener("blur", save);

  const el = h("article", { class: `card${rating ? ` rated-${rating}` : ""}`, tabindex: "0", "data-qid": q.id },
    h("div", { class: "card-head" },
      h("span", { class: "qnum" }, `#${num}`),
      h("span", { class: "pill" }, typeLabel[q.type] || q.type),
      q.difficulty ? h("span", { class: `pill ${q.difficulty}` }, q.difficulty) : null,
      g.strategy && g.strategy !== q.type ? h("span", { class: "pill", title: "generation strategy" }, g.strategy.replace(/_/g, " ")) : null,
      ev.selection_rank ? h("span", { class: "pill", title: "rank in best-of selection" }, `rank ${ev.selection_rank}`) : null),
    h("p", { class: "question" }, q.question), options, answer, more,
    h("div", { class: "feedback" }, h("div", { class: "thumbs" }, up, down), comment, saveState));
  el._actions = { up: () => { rating = rating === "up" ? null : "up"; save(); }, down: () => { rating = rating === "down" ? null : "down"; save(); },
    comment: () => comment.focus(), reveal: () => { if (hidden) { hidden = false; el.replaceWith(rebuild(false)); } } };
  function rebuild(hide) { const prev = state.hideAnswers; state.hideAnswers = hide; const c = card(q, num); state.hideAnswers = prev; return c; }
  return el;
}

/* ------------------------------------------------------------------ keyboard */
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const t = e.target;
  if (t.closest && t.closest("input, textarea, select")) {
    if (e.key === "Escape") t.blur();
    return;
  }
  const cards = [...document.querySelectorAll(".card")];
  if (!cards.length) return;
  const cur = t.closest ? t.closest(".card") : null;
  const i = cur ? cards.indexOf(cur) : -1;
  const focus = (j) => { const c = cards[Math.max(0, Math.min(cards.length - 1, j))]; c.focus(); c.scrollIntoView({ block: "nearest", behavior: "smooth" }); };
  const act = cur && cur._actions;
  switch (e.key) {
    case "j": focus(i + 1); break;
    case "k": focus(i < 0 ? 0 : i - 1); break;
    case "u": if (act) act.up(); else return; break;
    case "d": if (act) act.down(); else return; break;
    case "c": if (act) act.comment(); else return; break;
    case "a": if (act) { const qid = cur.dataset.qid; act.reveal(); const n = document.querySelector(`.card[data-qid="${CSS.escape(qid)}"]`); if (n) n.focus(); } else return; break;
    default: return;
  }
  e.preventDefault();
});

/* ------------------------------------------------------------------ routing */
async function route() {
  const m = location.hash.match(/^#\/run\/(.+)$/);
  if (m) return openRun(decodeURIComponent(m[1]));
  stopPoll();
  state.run = null;
  renderRunList();
  if (location.hash === "#/new" || !state.runs.length) return renderNew();
  main.replaceChildren(h("div", { class: "empty" }, h("p", {}, "Pick a run on the left to review its questions, or start a new one."),
    h("a", { class: "btn primary", href: "#/new" }, "+ New run")));
}

async function init() {
  const reviewer = $("#reviewer");
  reviewer.value = store("trivium.reviewer") || "";
  reviewer.addEventListener("change", () => store("trivium.reviewer", reviewer.value.trim()));
  state.hideAnswers = store("trivium.hideAnswers") === "1";
  try {
    state.settings = await api("/api/settings");
  } catch (e) {
    main.replaceChildren(h("div", { class: "empty error" }, `Cannot reach the trivium server: ${e.message}`));
    return;
  }
  $("#workspace").textContent = `Workspace: ${state.settings.workspace}`;
  await loadRuns();
  window.addEventListener("hashchange", route);
  route();
}

init();
