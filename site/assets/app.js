// metalworking glossary — client behavior (original code)
(function () {
  var doc = document, body = doc.body;

  // ---- theme switcher ----
  var themeBtns = doc.querySelectorAll(".themes button");
  function setTheme(t) {
    doc.documentElement.dataset.theme = t;
    try { localStorage.setItem("mw-theme", t); } catch (e) {}
    themeBtns.forEach(function (b) { b.classList.toggle("on", b.dataset.theme === t); });
  }
  themeBtns.forEach(function (b) {
    b.addEventListener("click", function () { setTheme(b.dataset.theme); });
  });
  setTheme(doc.documentElement.dataset.theme || "light");

  // ---- sidebar fold state ----
  var folds = {};
  try { folds = JSON.parse(localStorage.getItem("mw-folds") || "{}"); } catch (e) {}
  doc.querySelectorAll(".toc-section").forEach(function (sec) {
    var key = sec.dataset.sec;
    var hasActive = !!sec.querySelector("a.active");
    if (folds[key] && !hasActive) sec.classList.add("folded");
    var btn = sec.querySelector(".fold");
    function paint() { btn.innerHTML = sec.classList.contains("folded") ? "+" : "&minus;"; }
    paint();
    sec.querySelector(".toc-head").addEventListener("click", function () {
      sec.classList.toggle("folded");
      folds[key] = sec.classList.contains("folded");
      try { localStorage.setItem("mw-folds", JSON.stringify(folds)); } catch (e) {}
      paint();
    });
  });

  // ---- mobile burger ----
  var burger = doc.querySelector(".burger"), toc = doc.querySelector(".toc");
  if (burger) burger.addEventListener("click", function () { toc.classList.toggle("open"); });

  // ---- arrow-key navigation ----
  doc.addEventListener("keydown", function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (doc.activeElement && /INPUT|TEXTAREA/.test(doc.activeElement.tagName)) return;
    if (e.key === "ArrowRight" && body.dataset.next) location.href = body.dataset.next;
    if (e.key === "ArrowLeft" && body.dataset.prev) location.href = body.dataset.prev;
  });

  // ---- search ----
  var overlay = doc.getElementById("search-overlay");
  var input = doc.getElementById("search-input");
  var results = doc.getElementById("search-results");
  var index = null, sel = 0;

  function openSearch() {
    overlay.hidden = false;
    input.value = ""; results.innerHTML = ""; sel = 0;
    input.focus();
    if (!index) fetch("/search.json").then(function (r) { return r.json(); }).then(function (j) { index = j; });
  }
  function closeSearch() { overlay.hidden = true; }

  doc.getElementById("search-open") &&
    doc.getElementById("search-open").addEventListener("click", openSearch);
  overlay.addEventListener("click", function (e) { if (e.target === overlay) closeSearch(); });
  doc.addEventListener("keydown", function (e) {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openSearch(); }
    if (e.key === "Escape" && !overlay.hidden) closeSearch();
  });

  function paintResults(items) {
    results.innerHTML = items.slice(0, 12).map(function (it, i) {
      return '<li' + (i === sel ? ' class="sel"' : "") + '><a href="' + it.route + '">' +
        it.title + '<span class="sec">' + it.section + "</span></a></li>";
    }).join("");
  }
  function currentMatches() {
    var q = input.value.trim().toLowerCase();
    if (!index || !q) return [];
    return index.filter(function (it) {
      return (it.title + " " + it.section + " " + it.text).toLowerCase().indexOf(q) !== -1;
    });
  }
  input.addEventListener("input", function () { sel = 0; paintResults(currentMatches()); });
  input.addEventListener("keydown", function (e) {
    var items = currentMatches();
    if (e.key === "ArrowDown") { sel = Math.min(sel + 1, Math.min(items.length, 12) - 1); paintResults(items); e.preventDefault(); }
    if (e.key === "ArrowUp") { sel = Math.max(sel - 1, 0); paintResults(items); e.preventDefault(); }
    if (e.key === "Enter" && items[sel]) location.href = items[sel].route;
    e.stopPropagation();
  });
})();
