/* Open the GitHub repo link (top-right header icon) in a new tab */
document.addEventListener("click", function (e) {
  var source = e.target.closest(".md-source");
  if (source) {
    e.preventDefault();
    e.stopPropagation();
    window.open(source.href, "_blank", "noopener,noreferrer");
  }
});
