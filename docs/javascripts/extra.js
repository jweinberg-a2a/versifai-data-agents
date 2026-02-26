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
  var mermaid = e.target.closest(".mermaid");
  if (!mermaid) return;

  var svg = mermaid.querySelector("svg");
  if (!svg) return;

  var overlay = document.createElement("div");
  overlay.className = "mermaid-overlay";

  var closeBtn = document.createElement("button");
  closeBtn.className = "mermaid-overlay-close";
  closeBtn.textContent = "Esc to close";

  var clone = svg.cloneNode(true);
  clone.removeAttribute("width");
  clone.removeAttribute("height");
  clone.style.width = "auto";
  clone.style.height = "auto";

  overlay.appendChild(closeBtn);
  overlay.appendChild(clone);
  document.body.appendChild(overlay);

  function close() {
    overlay.remove();
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
