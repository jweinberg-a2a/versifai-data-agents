/* Open the GitHub repo link (top-right header icon) in a new tab */
document.addEventListener("DOMContentLoaded", function () {
  var source = document.querySelector(".md-header .md-source");
  if (source) {
    source.setAttribute("target", "_blank");
    source.setAttribute("rel", "noopener noreferrer");
  }
});

/* Expand Mermaid diagrams on click — fullscreen overlay */
document.addEventListener("click", function (e) {
  /* Already in overlay — ignore */
  if (e.target.closest(".mermaid-overlay")) return;

  /* Find the SVG: could be inside .mermaid, pre.mermaid, or have a mermaid ID */
  var svg = null;
  var container =
    e.target.closest(".mermaid") ||
    e.target.closest("pre.mermaid") ||
    e.target.closest('[class*="mermaid"]');

  if (container) {
    svg = container.querySelector("svg");
  }

  /* Fallback: clicked directly on an SVG with a mermaid ID */
  if (!svg) {
    var el = e.target.closest("svg");
    if (el && el.id && el.id.indexOf("mermaid") !== -1) {
      svg = el;
    }
  }

  if (!svg) return;

  var overlay = document.createElement("div");
  overlay.className = "mermaid-overlay";

  var closeBtn = document.createElement("button");
  closeBtn.className = "mermaid-overlay-close";
  closeBtn.textContent = "Esc to close";

  var clone = svg.cloneNode(true);
  clone.removeAttribute("width");
  clone.removeAttribute("height");
  clone.removeAttribute("style");
  clone.setAttribute("width", "100%");
  clone.setAttribute("height", "100%");

  overlay.appendChild(closeBtn);
  overlay.appendChild(clone);
  document.body.appendChild(overlay);
  document.body.style.overflow = "hidden";

  function close() {
    overlay.remove();
    document.body.style.overflow = "";
  }

  overlay.addEventListener("click", close);
  closeBtn.addEventListener("click", function (ev) {
    ev.stopPropagation();
    close();
  });
  document.addEventListener("keydown", function handler(ev) {
    if (ev.key === "Escape") {
      close();
      document.removeEventListener("keydown", handler);
    }
  });
});
