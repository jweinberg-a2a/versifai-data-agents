/* Open the GitHub repo link (top-right header icon) in a new tab */
document.addEventListener("DOMContentLoaded", function () {
  var source = document.querySelector(".md-header .md-source");
  if (source) {
    source.setAttribute("target", "_blank");
    source.setAttribute("rel", "noopener noreferrer");
  }
});
