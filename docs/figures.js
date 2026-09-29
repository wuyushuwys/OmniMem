// Renders the paper figures from docs/figures.yaml into #figure-gallery.

(async function renderFigures() {
  const mount = document.getElementById("figure-gallery");
  if (!mount) return;

  // ---- load config -------------------------------------------------------
  let cfg;
  try {
    const res = await fetch("figures.yaml", { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    cfg = jsyaml.load(await res.text());
  } catch (err) {
    const p = document.createElement("p");
    p.className = "gallery-error";
    p.textContent =
      `Could not load figures.yaml (${err.message}). If you opened this page as a ` +
      `local file, serve it over http instead, e.g. "python3 -m http.server".`;
    mount.appendChild(p);
    return;
  }

  const base = (cfg.assets_base || "").replace(/\/+$/, "");
  const sections = Array.isArray(cfg.sections) ? cfg.sections : [];
  const url = (src) => (base && src ? `${base}/${src}` : src || "");

  function el(tag, className, attrs) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (attrs) for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  }

  // ---- build sections ----------------------------------------------------
  sections.forEach((sec) => {
    if (!sec) return;
    const section = el("section", "figure-section", sec.id ? { id: sec.id } : null);

    if (sec.title) {
      const h2 = el("h2");
      h2.textContent = sec.title;
      section.appendChild(h2);
    }
    if (sec.animation) {
      const player = el("div", "method-film-player");
      const frame = el("iframe", "", {
        src: sec.animation.src,
        title: sec.animation.title || `${sec.title} animation`,
        allow: "autoplay; fullscreen",
        allowfullscreen: "",
      });
      // The embedded player reports its responsive height when its layout changes.
      window.addEventListener("message", (event) => {
        if (event.origin !== window.location.origin || event.source !== frame.contentWindow) return;
        if (event.data?.kind !== "omnimem-height") return;
        const height = Number(event.data.height);
        if (Number.isFinite(height) && height >= 200 && height <= 4000) {
          frame.style.height = Math.ceil(height) + "px";
        }
      });
      player.appendChild(frame);
      section.appendChild(player);
      const links = el("div", "method-film-links");
      (sec.animation.links || []).forEach((item) => {
        const link = el("a", "", { href: item.href });
        link.textContent = item.label;
        if (item.download) link.setAttribute("download", "");
        else {
          link.target = "_blank";
          link.rel = "noopener";
        }
        links.appendChild(link);
      });
      section.appendChild(links);
    }
    const images = Array.isArray(sec.images) ? sec.images : [];
    // Default: everything in one row. `cols:` overrides; clamp to the grid's 1–4.
    const cols = Math.min(Math.max(parseInt(sec.cols, 10) || images.length || 1, 1), 4);
    const grid = el("div", `grid cols-${cols}`);
    if (sec.equal_height) grid.classList.add("equal-height"); // same height, widths vary

    images.forEach((im) => {
      if (!im) return;
      const fig = el("figure", "figure");
      const img = el("img");
      img.src = url(im.src);
      img.alt = im.alt || im.caption || im.src || "";
      img.loading = "lazy";
      // Show a labeled placeholder if the image file isn't present yet.
      img.addEventListener("error", () => img.classList.add("img-missing"));
      // Equal-height row: width each figure by its aspect ratio.
      if (sec.equal_height) {
        const setRatio = () => {
          if (img.naturalWidth && img.naturalHeight) {
            fig.style.flexGrow = (img.naturalWidth / img.naturalHeight).toFixed(4);
          }
        };
        if (img.complete) setRatio();
        else img.addEventListener("load", setRatio);
      }
      fig.appendChild(img);
      if (im.caption) {
        const cap = el("figcaption");
        cap.textContent = im.caption;
        fig.appendChild(cap);
      }
      grid.appendChild(fig);
    });

    if (images.length) section.appendChild(grid);

    // Body / description renders BELOW the images.
    if (sec.body) {
      const body = el("p", "figure-body");
      body.textContent = sec.body;
      section.appendChild(body);
    }

    mount.appendChild(section);
  });
  // Resolve the initial fragment after the configured sections have been mounted.
  if (window.location.hash === "#framework") {
    document.getElementById("framework")?.scrollIntoView({ block: "start" });
  }
})();
