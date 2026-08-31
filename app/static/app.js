const PAGES = [
  ["overview", "Overview"],
  ["collect", "Data Collection"],
  ["explorer", "Feedback Explorer"],
  ["intents", "Wishlist Motivations"],
  ["barriers", "Purchase Barriers"],
  ["uncertainties", "Uncertainties"],
  ["roots", "Root Causes"],
  ["segments", "User Segments"],
  ["themes", "Theme Clusters"],
  ["seeking", "External Info Seeking"],
  ["opportunities", "Opportunity Matrix"],
  ["evidence", "Evidence Explorer"],
  ["report", "Discovery Report"],
];

const SUBS = {
  overview: "Source integrity, volume, and conversion-relevant signals.",
  collect: "Collect public Play Store and App Store reviews. Identity is validated first.",
  explorer: "Original review text beside AI analysis. Quotes are never generated.",
  intents: "Why users appear to save, browse, or delay — only where evidenced.",
  barriers: "Purchase blockers discovered from reviews, not a preset list.",
  uncertainties: "Unanswered questions that sit between interest and purchase.",
  roots: "Observed vs inferred vs hypothesized problem statements.",
  segments: "Behavioral groups that the evidence actually supports.",
  themes: "Emergent clusters. Taxonomy is not fixed in advance.",
  seeking: "When users look outside the app for information.",
  opportunities: "Deterministic scores: reach × frequency × impact × severity × confidence.",
  evidence: "Every insight must open the underlying source records.",
  report: "Full discovery write-up. No product solution is proposed.",
};

const view = document.getElementById("view");
const nav = document.getElementById("nav");
const title = document.getElementById("page-title");
const sub = document.getElementById("page-sub");
const banners = document.getElementById("source-banners");
const myntraOnly = document.getElementById("myntra-only");
const drawer = document.getElementById("drawer");
const drawerTitle = document.getElementById("drawer-title");
const drawerBody = document.getElementById("drawer-body");

PAGES.forEach(([id, label]) => {
  const a = document.createElement("a");
  a.href = `#${id}`;
  a.textContent = label;
  a.dataset.page = id;
  nav.appendChild(a);
});

function pageId() {
  return (location.hash || "#overview").slice(1);
}

function myntraParam() {
  return myntraOnly.checked ? "true" : "false";
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function chipFor(review) {
  if (review.is_synthetic) {
    return `<span class="chip amber">SYNTHETIC DEMONSTRATION DATA — NOT REAL USER DATA</span>`;
  }
  if (!review.is_valid_source) {
    return `<span class="chip amber">REFERENCE / NON-MYNTRA DATA</span>`;
  }
  return `<span class="chip green">MYNTRA EVIDENCE</span>`;
}

function bar(pct) {
  const width = Math.max(0, Math.min(100, Number(pct) || 0));
  return `<div class="bar"><span style="width:${width}%"></span></div>`;
}

function empty(msg) {
  return `<div class="card empty">${esc(msg)}</div>`;
}

async function loadBanners() {
  try {
    const data = await api("/api/sources");
    const cfg = await api("/api/config");
    const parts = [];
    parts.push(
      `<div class="banner">Configured Google Play app ID: <strong>${esc(cfg.google_play.app_id)}</strong> · Apple app ID: <strong>${esc(cfg.apple.app_id)}</strong> (${esc(cfg.apple.primary_region)} → fallback ${esc(cfg.apple.fallback_region)}) · AI: ${esc(cfg.ai_provider)} / ${esc(cfg.ai_model)} ${cfg.ai_configured ? "" : "· <strong>API key not set</strong>"}</div>`
    );
    (data.collected || []).forEach((src) => {
      const cls = src.is_valid_for_myntra ? "ok" : "warn";
      const msg =
        src.warning ||
        `${src.platform}: ${src.detected_app_name || "unknown"} (${src.validation_status})`;
      parts.push(
        `<div class="banner ${cls}"><strong>${esc(src.platform)}</strong> · Configured ID: ${esc(src.app_id)} · Detected app: ${esc(src.detected_app_name || "—")} · Developer: ${esc(src.detected_developer || "—")} · Status: ${esc(src.validation_status)}<br/>${esc(msg)}</div>`
      );
    });
    banners.innerHTML = parts.join("");
  } catch (err) {
    banners.innerHTML = `<div class="banner bad">${esc(err.message)}</div>`;
  }
}

function renderOverview(data) {
  const srcRows = (data.sources || [])
    .map(
      (s) => `<tr>
        <td>${esc(s.platform)}</td>
        <td>${esc(s.app_id)}</td>
        <td>${esc(s.detected_app_name)}</td>
        <td>${esc(s.detected_developer)}</td>
        <td>${esc(s.region || "—")}</td>
        <td>${esc(s.validation_status)}</td>
        <td>${s.review_count}</td>
      </tr>`
    )
    .join("");
  const run = data.latest_run;
  return `
    <div class="grid cols-4">
      <div class="card"><h3>Total reviews</h3><div class="stat">${data.total_reviews}</div><div class="stat-sub">stored, non-empty</div></div>
      <div class="card"><h3>Myntra-valid</h3><div class="stat">${data.myntra_reviews}</div><div class="stat-sub">eligible as Myntra evidence</div></div>
      <div class="card"><h3>Reference / non-Myntra</h3><div class="stat">${data.reference_non_myntra_reviews}</div><div class="stat-sub">must not be labelled Myntra</div></div>
      <div class="card"><h3>Relevant (analyzed)</h3><div class="stat">${data.relevant_reviews}</div><div class="stat-sub">${data.relevant_pct_of_analyzed}% of analyzed</div></div>
    </div>
    <div class="grid cols-2" style="margin-top:14px">
      <div class="card">
        <h3>Latest collection run</h3>
        ${
          run
            ? `<p>Status <strong>${esc(run.status)}</strong> · fetched ${run.fetched} · valid ${run.valid} · rejected ${run.rejected} · duplicates ${run.duplicates} · new ${run.new} · analyzed ${run.analyzed} · ${run.duration_seconds}s</p>
               <p class="muted">${(run.errors || []).join(" · ") || "No errors recorded."}</p>`
            : `<p class="muted">No collection run yet. Use Collect New Data.</p>`
        }
      </div>
      <div class="card">
        <h3>Signals</h3>
        <p>Wishlist ${data.signals?.wishlist_signal || 0} (${data.signals?.wishlist_pct || 0}%)</p>
        <p>Purchase hesitation ${data.signals?.purchase_hesitation || 0} (${data.signals?.hesitation_pct || 0}%)</p>
        <p class="muted">${esc(data.signals?.note || "")}</p>
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <h3>Source validation ledger</h3>
      <table class="table">
        <thead><tr><th>Source</th><th>App ID</th><th>Detected app</th><th>Developer</th><th>Region</th><th>Status</th><th>Reviews</th></tr></thead>
        <tbody>${srcRows || `<tr><td colspan="7" class="muted">No sources collected yet.</td></tr>`}</tbody>
      </table>
    </div>`;
}

function renderCollect(cfg, runs, status) {
  const gp = cfg.google_play;
  const ap = cfg.apple;
  const gpStatus = (status && status.google_play) || {};
  const apStatus = (status && status.apple_app_store) || {};
  const gpPass = gpStatus.validation === "PASS";
  const apPass = apStatus.validation === "PASS";
  const runRows = (runs || [])
    .map(
      (r) => `<tr>
        <td>${r.id}</td><td>${esc(r.status)}</td><td>${esc(r.sources)}</td>
        <td>${r.fetched}</td><td>${r.valid}</td><td>${r.rejected}</td>
        <td>${r.duplicates}</td><td>${r.new}</td><td>${r.analyzed}</td>
        <td>${r.duration_seconds}s</td>
      </tr>`
    )
    .join("");
  return `
    <div class="grid cols-2">
      <div class="card">
        <h3>Google Play</h3>
        <p>App: <strong>${esc(gp.app || "Myntra")}</strong></p>
        <p>Package: <strong>${esc(gp.package || gp.app_id)}</strong></p>
        <p>URL: <a href="${esc(gp.url || gp.expected_url)}" target="_blank" rel="noopener">${esc(gp.url || gp.expected_url)}</a></p>
        <p>Configured ID: ${esc(gp.app_id)}</p>
        <p>Detected App: ${esc(gpStatus.detected_app || "—")}</p>
        <p>Expected App: ${esc(gp.expected_app)}</p>
        <p>Validation: <span class="chip ${gpPass ? "green" : "amber"}">${esc(gpStatus.validation || "FAIL")}</span></p>
        <p>Reviews collected: <strong>${gpStatus.reviews_collected ?? 0}</strong></p>
        <p>New reviews (last run): ${gpStatus.new_reviews ?? 0} · Duplicates: ${gpStatus.duplicates ?? 0}</p>
        <p>Last collection: ${esc(gpStatus.last_collection || "never")}</p>
        ${gpStatus.warning ? `<p class="muted">${esc(gpStatus.warning)}</p>` : ""}
        <button class="btn" data-collect="google_play">Collect Google Play Reviews</button>
      </div>
      <div class="card">
        <h3>Apple App Store</h3>
        <p>App: <strong>${esc(ap.app || "Myntra Fashion Shopping App")}</strong></p>
        <p>App ID: <strong>${esc(ap.app_id)}</strong></p>
        <p>URL: <a href="${esc(ap.url || ap.expected_url)}" target="_blank" rel="noopener">${esc(ap.url || ap.expected_url)}</a></p>
        <p>Configured ID: ${esc(ap.app_id)}</p>
        <p>Detected App: ${esc(apStatus.detected_app || "—")}</p>
        <p>Expected App: ${esc(ap.expected_app)}</p>
        <p>Primary region: India · Fallback region: US</p>
        <p>Validation: <span class="chip ${apPass ? "green" : "amber"}">${esc(apStatus.validation || "FAIL")}</span></p>
        <p>Reviews collected: <strong>${apStatus.reviews_collected ?? 0}</strong></p>
        <p>New reviews (last run): ${apStatus.new_reviews ?? 0} · Duplicates: ${apStatus.duplicates ?? 0}</p>
        <p>Last collection: ${esc(apStatus.last_collection || "never")}</p>
        ${apStatus.warning ? `<p class="muted">${esc(apStatus.warning)}</p>` : ""}
        <button class="btn" data-collect="apple_app_store">Collect Apple App Store Reviews</button>
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <h3>Collect all</h3>
      <label>Max reviews per source <input id="max-reviews" type="number" min="10" max="5000" value="50" /></label>
      <label style="margin-left:12px"><input type="checkbox" id="do-analyze" checked /> Run AI analysis on new reviews</label>
      <div style="margin-top:12px" class="top-actions">
        <button class="btn primary" data-collect="all">Collect All</button>
        <button class="btn" id="btn-analyze">Analyze existing</button>
      </div>
      <p class="muted" style="margin-top:10px">CSV/JSON upload is a fallback only. Default max is 50 per run so a first collection stays manageable; the configured ceiling is ${gp.max_reviews}.</p>
      <input type="file" id="upload-file" accept=".csv,.json" />
      <label><input type="checkbox" id="upload-synthetic" /> Mark as synthetic demo data</label>
      <button class="btn" id="btn-upload">Upload fallback</button>
      <div id="collect-log" class="log" style="margin-top:12px">Ready.</div>
    </div>
    <div class="card" style="margin-top:14px">
      <h3>Collection history</h3>
      <table class="table">
        <thead><tr><th>ID</th><th>Status</th><th>Sources</th><th>Fetched</th><th>Valid</th><th>Rejected</th><th>Dupes</th><th>New</th><th>Analyzed</th><th>Duration</th></tr></thead>
        <tbody>${runRows || `<tr><td colspan="10">No runs yet.</td></tr>`}</tbody>
      </table>
    </div>`;
}

function reviewCard(r) {
  const a = r.analysis;
  return `<article class="card" data-open-review="${r.id}" style="margin-bottom:10px">
    <div>${chipFor(r)} <span class="chip">${esc(r.source)}</span> <span class="chip">★ ${r.rating ?? "—"}</span> ${r.region ? `<span class="chip">${esc(r.region)}</span>` : ""}</div>
    <p class="quote">${esc(r.title ? r.title + " — " : "")}${esc(r.text)}</p>
    <p class="meta">${esc(r.app_name)} · ${esc(r.app_id)} · ${esc(r.source_review_id)} · ${esc(r.review_date || r.collected_at || "")} · ${esc(r.source_url)}</p>
    ${r.warning ? `<p class="muted">${esc(r.warning)}</p>` : ""}
    ${
      a
        ? `<p><strong>Observed/inferred root:</strong> ${esc(a.root_cause?.statement || "")}</p>
           <p class="muted">Intent: ${(a.intent || []).map(esc).join(", ") || "—"} · Barriers: ${(a.barriers || []).map(esc).join(", ") || "—"} · Uncertainties: ${(a.uncertainties || []).map(esc).join(", ") || "—"}</p>
           <p class="muted">Wishlist ${esc(a.wishlist_signal)} · Purchase ${esc(a.purchase_signal)} · Hesitation ${esc(a.purchase_hesitation)} · Confidence ${a.confidence}/5</p>`
        : `<p class="muted">Not yet analyzed.</p>`
    }
  </article>`;
}

function renderExplorer(payload) {
  const items = payload.items || [];
  return `
    <div class="filters">
      <input id="q" placeholder="Search original text" />
      <select id="f-source"><option value="">All sources</option><option value="google_play">Google Play</option><option value="apple_app_store">App Store</option></select>
      <select id="f-rating"><option value="">All ratings</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select>
      <input id="f-barrier" placeholder="Barrier" />
      <input id="f-intent" placeholder="Intent" />
      <input id="f-unc" placeholder="Uncertainty" />
      <input id="f-theme" placeholder="Theme" />
      <select id="f-wish"><option value="">Wishlist signal</option><option>explicit</option><option>implicit</option><option>none</option></select>
      <select id="f-purch"><option value="">Purchase signal</option><option>purchased</option><option>intend_to_purchase</option><option>hesitant</option><option>abandoned</option><option>none</option></select>
      <button class="btn" id="apply-filters">Apply filters</button>
    </div>
    <p class="muted">${payload.total} matching reviews</p>
    ${items.map(reviewCard).join("") || empty("No reviews yet. Collect data first.")}
  `;
}

function distTable(rows, kind) {
  if (!rows.length) return empty("No analyzed labels yet. Collect and run AI analysis.");
  return `<table class="table">
    <thead><tr><th>Label</th><th>Count</th><th>% of relevant</th><th>Sources</th><th>Hesitant</th></tr></thead>
    <tbody>
      ${rows
        .map(
          (r) => `<tr data-ids="${esc((r.review_ids || []).join(","))}" data-insight="${esc(r.label)}">
            <td>${esc(r.label)}</td>
            <td>${r.count}</td>
            <td>${r.percentage}% of ${r.denominator}<div style="margin-top:6px">${bar(r.percentage)}</div></td>
            <td>${(r.sources || []).map(esc).join(", ")}</td>
            <td>${r.hesitant_count}</td>
          </tr>`
        )
        .join("")}
    </tbody>
  </table>`;
}

function renderListPage(titleText, rows, note) {
  return `<div class="card"><h3>${esc(titleText)}</h3><p class="muted">${esc(note)}</p>${distTable(rows)}</div>`;
}

function renderOpportunities(rows) {
  if (!rows.length) return empty("No opportunities scored yet.");
  return `
    <p class="muted">Score = Reach × Frequency × Purchase Impact × Severity × Evidence Confidence (each 1–5). Calculated in code, not by the LLM.</p>
    <table class="table">
      <thead><tr><th>#</th><th>Problem</th><th>Score</th><th>R</th><th>F</th><th>I</th><th>S</th><th>C</th><th>%</th><th>Sources</th><th>Status</th></tr></thead>
      <tbody>
        ${rows
          .map(
            (o) => `<tr data-kind="opportunity" data-id="${o.id}">
              <td>${o.rank}</td>
              <td>${esc(o.user_problem)} ${o.includes_non_myntra ? '<span class="chip amber">includes non-Myntra</span>' : ""}</td>
              <td class="score">${o.score}</td>
              <td>${o.reach}</td><td>${o.frequency}</td><td>${o.purchase_impact}</td><td>${o.severity}</td><td>${o.evidence_confidence}</td>
              <td>${o.percentage}% (${o.relevant_count}/${o.total_relevant})</td>
              <td>${(o.sources || []).map(esc).join(", ")}</td>
              <td>${esc(o.cross_source_status)}</td>
            </tr>`
          )
          .join("")}
      </tbody>
    </table>`;
}

function renderNamed(rows, kind) {
  if (!rows.length) return empty("Nothing discovered yet.");
  return rows
    .map(
      (t) => `<div class="card" data-kind="${kind}" data-id="${t.id}" style="margin-bottom:10px;cursor:pointer">
        <h3>${esc(t.name)}</h3>
        <p>${esc(t.description || "")}</p>
        <p class="muted">${t.review_count} reviews · Myntra ${t.myntra_review_count || 0} · sources ${(t.sources || []).join(", ")}</p>
      </div>`
    )
    .join("");
}

function renderSeeking(rows) {
  if (!rows.length) return empty("No external information-seeking mentions extracted yet.");
  return rows
    .map(
      (s) => `<div class="card" style="margin-bottom:10px">
        <h3>${esc(s.source)} · ${s.count} (${s.percentage}% of ${s.denominator})</h3>
        ${(s.examples || [])
          .slice(0, 5)
          .map(
            (ex) =>
              `<p class="muted">Review #${ex.review_id}: what=${esc(ex.what || "")} why=${esc(ex.why || "")} hesitation=${ex.associated_with_hesitation} lack=${ex.myntra_appears_to_lack_info} basis=${esc(ex.basis)} ${ex.is_valid_source ? "" : "[non-Myntra]"}</p>`
          )
          .join("")}
      </div>`
    )
    .join("");
}

function renderReport(rep) {
  const top = rep.top_opportunities || [];
  const src = (rep.sections?.["3_data_sources"] || [])
    .map(
      (s) =>
        `<li>${esc(s.platform)} · ${esc(s.app_id)} · detected ${esc(s.detected_app_name)} · ${esc(s.validation_status)}</li>`
    )
    .join("");
  const topHtml = top
    .map(
      (o) => `<div class="card" style="margin-bottom:12px">
        <h3>#${o.rank} ${esc(o.opportunity)} · score ${o.opportunity_score}</h3>
        <p>${esc(o.user_problem)}</p>
        <p class="muted">${o.affected_users.relevant_count} reviews (${o.affected_users.percentage_of_relevant}% of ${o.affected_users.denominator}) · Myntra-only ${o.affected_users.myntra_only_count}</p>
        ${(o.evidence || [])
          .map(
            (e) =>
              `<blockquote class="quote">“${esc(e.quote)}”</blockquote><p class="meta">${esc(e.source)} · ${esc(e.app_name)} · ${esc(e.data_classification)} · ${esc(e.source_url)}</p>`
          )
          .join("")}
        <p><strong>What we know.</strong> ${esc(o.what_we_know)}</p>
        <p><strong>What we don’t know.</strong> ${esc(o.what_we_dont_know)}</p>
        <p><strong>Why investigate.</strong> ${esc(o.why_it_deserves_investigation)}</p>
      </div>`
    )
    .join("");
  const primary = rep.single_most_promising_problem || {};
  return `<article class="report">
    <div class="card">
      <h3>Executive summary</h3>
      <p>${esc(rep.research_question)}</p>
      <p>${esc(rep.business_goal)}</p>
      <p class="muted">${esc(rep.anti_solution_note)}</p>
    </div>
    <h2>1. Business goal</h2><p>${esc(rep.sections["1_business_goal"].goal)}</p>
    <h2>2. Data collection method</h2><p>${esc(rep.sections["2_data_collection_method"].primary)}</p><p>${esc(rep.sections["2_data_collection_method"].india_first)}</p>
    <h2>3. Data sources</h2><ul>${src}</ul>
    <h2>4. Data volume</h2><p>Total ${rep.sections["4_data_volume"].total_reviews} · Myntra-valid ${rep.sections["4_data_volume"].myntra_reviews} · Reference ${rep.sections["4_data_volume"].reference_non_myntra_reviews} · Analyzed ${rep.sections["4_data_volume"].analyzed_reviews}</p>
    <h2>5. Data quality</h2><ul>${(rep.sections["5_data_quality"] || []).map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
    <h2>6–8. Motivations, barriers, uncertainties</h2>
    <p class="muted">See dedicated dashboard pages. Counts are programmatic (count / relevant analyzed × 100).</p>
    <h2>15. Opportunity scoring & Top 5</h2>
    ${topHtml || empty("No scored opportunities.")}
    <h2>The single most promising user problem</h2>
    <div class="card"><p>${esc(primary.why_first || "")}</p></div>
    <h2>19. What we know</h2><ul>${(rep.sections["19_what_we_know"] || []).map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
    <h2>20. What we don’t know</h2><ul>${(rep.sections["20_what_we_dont_know"] || []).map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
    <h2>21. Recommended next research steps</h2><ul>${(rep.sections["21_recommended_next_research_steps"] || []).map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
  </article>`;
}

async function openEvidence(kind, id, titleText) {
  drawer.hidden = false;
  drawerTitle.textContent = titleText || "Evidence";
  drawerBody.innerHTML = "<p class='muted'>Loading source records…</p>";
  try {
    const data = await api(`/api/evidence/${kind}/${id}`);
    drawerBody.innerHTML = (data.evidence || [])
      .map(
        (e) => `<div class="evidence-item">
          ${e.is_synthetic ? `<div class="chip amber">SYNTHETIC DEMONSTRATION DATA — NOT REAL USER DATA</div>` : ""}
          ${e.is_valid_source ? `<div class="chip green">MYNTRA EVIDENCE</div>` : `<div class="chip amber">REFERENCE / NON-MYNTRA DATA</div>`}
          <p class="quote">“${esc(e.quote)}”</p>
          <p class="meta">Source ${esc(e.source)} · App ${esc(e.app_name)} (${esc(e.app_id)}) · Review ${esc(e.source_review_id)} · ${esc(e.date || "")}</p>
          <p class="meta">${esc(e.source_url)}</p>
        </div>`
      )
      .join("") || "<p class='muted'>No linked source records.</p>";
  } catch (err) {
    drawerBody.innerHTML = `<p class="muted">${esc(err.message)}</p>`;
  }
}

async function openIds(ids, insight) {
  drawer.hidden = false;
  drawerTitle.textContent = insight || "Evidence";
  drawerBody.innerHTML = "";
  for (const id of ids.slice(0, 15)) {
    try {
      const r = await api(`/api/reviews/${id}`);
      drawerBody.insertAdjacentHTML("beforeend", reviewCard(r));
    } catch {
      /* skip */
    }
  }
}

function logCollect(message) {
  const el = document.getElementById("collect-log");
  if (!el) return;
  el.textContent += `\n${message}`;
  el.scrollTop = el.scrollHeight;
}

function collectStream(sources) {
  const max = document.getElementById("max-reviews")?.value || "";
  const analyze = document.getElementById("do-analyze")?.checked ?? true;
  const params = new URLSearchParams({
    sources,
    analyze: String(analyze),
  });
  if (max) params.set("max_reviews", max);
  const es = new EventSource(`/api/collect/stream?${params.toString()}`);
  logCollect(`Starting collection: ${sources}`);
  es.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      logCollect(JSON.stringify(data));
      if (data.stage === "done") {
        es.close();
        loadBanners();
      }
    } catch {
      logCollect(ev.data);
    }
  };
  es.onerror = () => {
    logCollect("Stream closed or failed. If collection is still running, wait and refresh.");
    es.close();
  };
}

async function render() {
  const id = pageId();
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === id));
  title.textContent = (PAGES.find((p) => p[0] === id) || ["", id])[1];
  sub.textContent = SUBS[id] || "";
  view.innerHTML = `<div class="card muted">Loading…</div>`;
  try {
    if (id === "overview") {
      view.innerHTML = renderOverview(await api(`/api/overview?myntra_only=${myntraParam()}`));
    } else if (id === "collect") {
      const [cfg, runs, status] = await Promise.all([
        api("/api/config"),
        api("/api/collection-runs"),
        api("/api/collection-status"),
      ]);
      view.innerHTML = renderCollect(cfg, runs, status);
    } else if (id === "explorer") {
      view.innerHTML = renderExplorer(await api(`/api/reviews?limit=40&myntra_only=${myntraParam()}`));
    } else if (id === "intents") {
      view.innerHTML = renderListPage(
        "Wishlist motivations",
        await api(`/api/intents?myntra_only=${myntraParam()}`),
        "Labels are extracted from reviews. Empty means the corpus did not evidence wishlist motives."
      );
    } else if (id === "barriers") {
      view.innerHTML = renderListPage(
        "Purchase barriers",
        await api(`/api/barriers?myntra_only=${myntraParam()}`),
        "Do not assume fit, price, or reviews are the largest problem until the counts say so."
      );
    } else if (id === "uncertainties") {
      view.innerHTML = renderListPage(
        "Uncertainties",
        await api(`/api/uncertainties?myntra_only=${myntraParam()}`),
        "Questions standing between interest and purchase."
      );
    } else if (id === "roots") {
      const opps = await api("/api/opportunities");
      view.innerHTML = `<div class="card"><h3>Root-cause ranked problems</h3>${renderOpportunities(opps)}</div>`;
    } else if (id === "segments") {
      view.innerHTML = renderNamed(await api("/api/segments"), "segment");
    } else if (id === "themes") {
      view.innerHTML = renderNamed(await api("/api/themes"), "theme");
    } else if (id === "seeking") {
      view.innerHTML = renderSeeking(await api(`/api/information-seeking?myntra_only=${myntraParam()}`));
    } else if (id === "opportunities") {
      view.innerHTML = `<div class="card">${renderOpportunities(await api("/api/opportunities"))}</div>`;
    } else if (id === "evidence") {
      const opps = await api("/api/opportunities");
      view.innerHTML = `<p class="muted">Click a problem to inspect original source records.</p><div class="card">${renderOpportunities(opps)}</div>`;
    } else if (id === "report") {
      view.innerHTML = renderReport(await api("/api/report"));
    } else {
      view.innerHTML = empty("Unknown page.");
    }
  } catch (err) {
    view.innerHTML = `<div class="banner bad">${esc(err.message)}</div>`;
  }
}

document.getElementById("drawer-close").onclick = () => {
  drawer.hidden = true;
};
document.getElementById("btn-collect-all").onclick = () => {
  location.hash = "#collect";
};
myntraOnly.onchange = render;
window.addEventListener("hashchange", render);

view.addEventListener("click", async (ev) => {
  const collectBtn = ev.target.closest("[data-collect]");
  if (collectBtn) {
    const which = collectBtn.dataset.collect;
    collectStream(which === "all" ? "google_play,apple_app_store" : which);
    return;
  }
  if (ev.target.id === "btn-analyze") {
    logCollect("Running analysis pipeline…");
    const res = await fetch("/api/analyze", { method: "POST" });
    logCollect(await res.text());
    return;
  }
  if (ev.target.id === "btn-upload") {
    const file = document.getElementById("upload-file").files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const syn = document.getElementById("upload-synthetic").checked;
    const res = await fetch(`/api/upload?is_synthetic=${syn}`, { method: "POST", body: fd });
    logCollect(await res.text());
    return;
  }
  if (ev.target.id === "apply-filters") {
    const params = new URLSearchParams({ limit: "40", myntra_only: myntraParam() });
    const q = document.getElementById("q").value;
    const source = document.getElementById("f-source").value;
    const rating = document.getElementById("f-rating").value;
    const barrier = document.getElementById("f-barrier").value;
    const intent = document.getElementById("f-intent").value;
    const unc = document.getElementById("f-unc").value;
    const theme = document.getElementById("f-theme").value;
    const wish = document.getElementById("f-wish").value;
    const purch = document.getElementById("f-purch").value;
    if (q) params.set("q", q);
    if (source) params.set("source", source);
    if (rating) params.set("rating", rating);
    if (barrier) params.set("barrier", barrier);
    if (intent) params.set("intent", intent);
    if (unc) params.set("uncertainty", unc);
    if (theme) params.set("theme", theme);
    if (wish) params.set("wishlist_signal", wish);
    if (purch) params.set("purchase_signal", purch);
    view.innerHTML = renderExplorer(await api(`/api/reviews?${params}`));
    return;
  }
  const kindRow = ev.target.closest("[data-kind]");
  if (kindRow) {
    openEvidence(kindRow.dataset.kind, kindRow.dataset.id, kindRow.textContent.trim().slice(0, 80));
    return;
  }
  const idsRow = ev.target.closest("[data-ids]");
  if (idsRow && idsRow.dataset.ids) {
    openIds(
      idsRow.dataset.ids.split(",").filter(Boolean).map(Number),
      idsRow.dataset.insight
    );
  }
});

loadBanners();
render();
