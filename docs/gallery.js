// Renders the video gallery from docs/videos.yaml.

(async function renderGallery() {
  const mount = document.getElementById("video-gallery");
  if (!mount) return;

  // ---- load config -------------------------------------------------------
  let cfg;
  try {
    const res = await fetch("videos.yaml", { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    cfg = jsyaml.load(await res.text());
  } catch (err) {
    const p = document.createElement("p");
    p.className = "gallery-error";
    p.textContent =
      `Could not load videos.yaml (${err.message}). If you opened this page as a ` +
      `local file, serve it over http instead, e.g. "python3 -m http.server".`;
    mount.appendChild(p);
    return;
  }

  const base = (cfg.assets_base || "").replace(/\/+$/, "");
  const sections = Array.isArray(cfg.sections) ? cfg.sections : [];
  const url = (src) => (base && src ? `${base}/${src}` : src || "");

  // ---- small DOM helpers -------------------------------------------------
  function el(tag, className, attrs) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (attrs) for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  }

  function videoEl(src, preload) {
    const v = el("video");
    v.src = url(src);
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    v.preload = preload || "metadata";
    return v;
  }

  // Prompt text shown as an overlay on hover, matching the LongLive page pattern.
  function promptText(item) {
    if (Array.isArray(item.segments) && item.segments.length) {
      return item.segments
        .map((s) => `${s.time || ""}: ${s.text || ""}`.trim())
        .join("\n");
    }
    return item.prompt || "";
  }

  function addPromptOverlay(media, item) {
    const text = promptText(item || {});
    if (!text) return;
    const overlay = el("div", "prompt-overlay");
    overlay.textContent = text;
    media.appendChild(overlay);
  }

  // A single video card with a hover prompt overlay.
  function mediaCard({ src, duration, label, ours, preload, promptItem }) {
    const card = el("div", "vcard");
    const media = el("div", "vmedia");
    if (label) {
      const cl = el("span", ours ? "col-label ours" : "col-label");
      cl.textContent = label;
      media.appendChild(cl);
    }
    if (duration) {
      const d = el("span", "duration-tag");
      d.textContent = duration;
      media.appendChild(d);
    }
    media.appendChild(videoEl(src, preload));
    addPromptOverlay(media, promptItem);
    card.appendChild(media);
    return card;
  }

  function speedBar(group) {
    const bar = el("div", "speed-bar", { "data-speed-group": group });
    const label = el("span");
    label.textContent = "Play speed:";
    bar.appendChild(label);
    [["1", "×1"], ["2", "×2"]].forEach(([rate, txt], i) => {
      const b = el("button", i === 0 ? "speed-btn active" : "speed-btn", { "data-speed": rate });
      b.textContent = txt;
      bar.appendChild(b);
    });
    return bar;
  }

  // ---- section renderers -------------------------------------------------
  let speedSeq = 0;
  let syncSeq = 0;

  // Side-by-side comparison rows; the two videos in a row start together.
  function renderCompare(sec) {
    const frag = document.createDocumentFragment();
    (sec.rows || []).forEach((row) => {
      const cr = el("div", "compare-row");
      if (row.title) {
        const t = el("p", "row-title");
        t.textContent = row.title;
        cr.appendChild(t);
      }
      const grid = el("div", "grid cols-2", { "data-sync-group": `sync${++syncSeq}` });
      const dur = row.duration || sec.duration;
      const L = row.left || {};
      const R = row.right || {};
      // Default: the right card is "Ours" (accent label); override with `ours:`.
      grid.appendChild(mediaCard({ src: L.src, duration: dur, label: L.label, ours: L.ours === true, preload: "auto", promptItem: row }));
      grid.appendChild(mediaCard({ src: R.src, duration: dur, label: R.label, ours: R.ours !== false, preload: "auto", promptItem: row }));
      cr.appendChild(grid);
      frag.appendChild(cr);
    });
    return frag;
  }

  // A grid of N videos, each with its own hover prompt overlay.
  function renderGrid(sec) {
    const cols = Math.min(Math.max(parseInt(sec.cols, 10) || 2, 1), 4);
    const grid = el("div", `grid cols-${cols}`);
    (sec.videos || []).forEach((item) => {
      const card = mediaCard({
        src: item.src,
        duration: item.duration || sec.duration,
        label: item.label,
        ours: item.ours === true,
        promptItem: item,
      });
      grid.appendChild(card);
    });
    return grid;
  }

  // ---- in-page nav (chips to each section) -------------------------------
  const navItems = sections.filter((s) => s && s.id && s.title);
  if (navItems.length > 1) {
    const nav = el("nav", "gallery-nav");
    navItems.forEach((s) => {
      const a = el("a");
      a.href = `#${s.id}`;
      a.textContent = s.title;
      nav.appendChild(a);
    });
    mount.appendChild(nav);
  }

  const upcoming = el("p", "gallery-upcoming");
  upcoming.textContent = "More videos from additional versions are coming soon.";
  mount.appendChild(upcoming);

  // ---- build sections ----------------------------------------------------
  sections.forEach((sec) => {
    if (!sec) return;
    const section = el("section", "video-section", sec.id ? { id: sec.id } : null);

    const h2 = el("h2");
    h2.textContent = sec.title || "";
    section.appendChild(h2);

    const hint = el("h3", "prompt-hint");
    hint.textContent = "Hover on the video to see corresponding text prompts";
    section.appendChild(hint);

    if (sec.lead) {
      const lead = el("p", "lead");
      lead.textContent = sec.lead;
      section.appendChild(lead);
    }

    // The speed toggle controls every video inside `target`.
    let target = section;
    if (sec.speed_toggle) {
      const group = `spd${++speedSeq}`;
      section.appendChild(speedBar(group));
      target = el("div", null, { "data-speed-group-target": group });
      section.appendChild(target);
    }

    target.appendChild(sec.type === "compare" ? renderCompare(sec) : renderGrid(sec));
    mount.appendChild(section);
  });

  // ---- behaviors ---------------------------------------------------------
  initSpeedBars();
  initPlayback();

  function initSpeedBars() {
    mount.querySelectorAll(".speed-bar").forEach((bar) => {
      const group = bar.getAttribute("data-speed-group");
      const region = mount.querySelector(`[data-speed-group-target="${group}"]`);
      if (!region) return;
      const buttons = bar.querySelectorAll(".speed-btn");
      buttons.forEach((btn) => {
        btn.addEventListener("click", () => {
          const rate = parseFloat(btn.getAttribute("data-speed"));
          buttons.forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          region.querySelectorAll("video").forEach((v) => { v.playbackRate = rate; });
        });
      });
    });
  }

  function initPlayback() {
    // Map each video to its sync group (videos that should co-start).
    const syncGroups = new Map();
    mount.querySelectorAll("[data-sync-group]").forEach((group) => {
      const vids = Array.from(group.querySelectorAll("video"));
      vids.forEach((v) => syncGroups.set(v, vids));
    });

    function waitReady(video) {
      return new Promise((resolve) => {
        if (video.readyState >= 3) { resolve(); return; }
        const on = () => { video.removeEventListener("canplay", on); resolve(); };
        video.addEventListener("canplay", on);
      });
    }

    async function playGroup(vids) {
      vids.forEach((v) => { try { v.pause(); } catch (_) {} });
      await Promise.all(vids.map(waitReady));
      vids.forEach((v) => { try { v.currentTime = 0; } catch (_) {} });
      vids.forEach((v) => v.play().catch(() => {}));
    }

    if (!("IntersectionObserver" in window)) {
      mount.querySelectorAll("video").forEach((v) => v.play().catch(() => {}));
      return;
    }

    const startedGroups = new WeakSet();
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          const v = e.target;
          const group = syncGroups.get(v);
          if (e.isIntersecting) {
            if (group) {
              if (!startedGroups.has(group)) { startedGroups.add(group); playGroup(group); }
              else v.play().catch(() => {});
            } else {
              v.play().catch(() => {});
            }
          } else {
            v.pause();
            if (group) startedGroups.delete(group); // re-sync next time it enters view
          }
        });
      },
      { rootMargin: "200px 0px" }
    );
    mount.querySelectorAll("video").forEach((v) => io.observe(v));
  }
})();
