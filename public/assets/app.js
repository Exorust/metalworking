// metalworking glossary — client behavior (original code)
(function () {
  var doc = document, body = doc.body;

  // ---- analytics events (Vercel Web Analytics queue) ----
  function track(name, data) {
    try { window.va && window.va("event", data ? { name: name, data: data } : { name: name }); } catch (e) {}
  }

  // ---- theme dropdown: a listbox with arrow keys, aria-selected, and focus that returns to the trigger ----
  var themeBtns = [].slice.call(doc.querySelectorAll(".theme-list button"));
  var themeOpen = doc.getElementById("theme-open");
  var themeList = doc.getElementById("theme-list");
  function setTheme(t) {
    doc.documentElement.dataset.theme = t;   // the button label follows this attribute through CSS
    try { localStorage.setItem("mw-theme", t); } catch (e) {}
    themeBtns.forEach(function (b) { b.setAttribute("aria-selected", String(b.dataset.theme === t)); });
  }
  function themeMenu(open, refocus) {
    if (!themeList) return;
    themeList.hidden = !open;
    themeOpen.setAttribute("aria-expanded", String(open));
    if (open) {
      var cur = themeList.querySelector('[aria-selected="true"]') || themeBtns[0];
      if (cur) cur.focus();
    } else if (refocus) themeOpen.focus();
  }
  if (themeOpen) {
    themeOpen.addEventListener("click", function (e) { e.stopPropagation(); themeMenu(themeList.hidden, false); });
    themeOpen.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); e.stopPropagation(); themeMenu(true, false); }
    });
  }
  themeBtns.forEach(function (b) {
    b.addEventListener("click", function () {
      setTheme(b.dataset.theme);
      track("theme_change", { theme: b.dataset.theme });
      themeMenu(false, true);
    });
  });
  if (themeList) themeList.addEventListener("keydown", function (e) {
    var i = themeBtns.indexOf(doc.activeElement), n = themeBtns.length;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault(); e.stopPropagation();
      themeBtns[(i + (e.key === "ArrowDown" ? 1 : n - 1)) % n].focus();
    } else if (e.key === "Home") { e.preventDefault(); themeBtns[0].focus(); }
    else if (e.key === "End") { e.preventDefault(); themeBtns[n - 1].focus(); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); themeMenu(false, true); }
    else if (e.key === "Tab") { themeMenu(false, false); }
  });
  doc.addEventListener("click", function (e) {
    if (themeList && !themeList.hidden && !themeList.contains(e.target)) themeMenu(false, false);
  });
  doc.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && themeList && !themeList.hidden) themeMenu(false, true);
  });
  setTheme(doc.documentElement.dataset.theme || "light");

  // ---- below 900px the search and theme controls live in the drawer ----
  var mq = matchMedia("(max-width: 900px)");
  var tools = doc.getElementById("toc-tools"), topBar = doc.querySelector(".top");
  var themes = doc.querySelector(".themes"), searchBtn = doc.getElementById("search-open");
  function placeTools() {
    if (!tools || !themes || !searchBtn) return;
    if (mq.matches) { tools.appendChild(searchBtn); tools.appendChild(themes); }
    else { var gh = topBar.querySelector(".gh-btn"); topBar.insertBefore(themes, gh); topBar.insertBefore(searchBtn, gh); }
  }
  placeTools();
  if (mq.addEventListener) mq.addEventListener("change", placeTools);

  // ---- sidebar fold state ----
  var folds = {};
  try { folds = JSON.parse(localStorage.getItem("mw-folds") || "{}"); } catch (e) {}
  doc.querySelectorAll(".toc-section").forEach(function (sec) {
    var key = sec.dataset.sec;
    var hasActive = !!sec.querySelector("a.active");
    if (folds[key] && !hasActive) sec.classList.add("folded");
    var btn = sec.querySelector(".fold");
    if (!btn) return;                      // the index link has no heading to fold
    function paint() {
      var folded = sec.classList.contains("folded");
      btn.innerHTML = folded ? "+" : "&minus;";
      btn.setAttribute("aria-expanded", String(!folded));
    }
    paint();
    sec.querySelector(".toc-head").addEventListener("click", function () {
      sec.classList.toggle("folded");
      folds[key] = sec.classList.contains("folded");
      try { localStorage.setItem("mw-folds", JSON.stringify(folds)); } catch (e) {}
      paint();
    });
  });

  // ---- sidebar: show the current page's row without scrolling the page ----
  var toc = doc.querySelector(".toc");
  function revealActive() {
    var active = toc && toc.querySelector("a.active");
    if (active) toc.scrollTop = Math.max(0, active.offsetTop - toc.clientHeight / 2);
  }
  if (!mq.matches) revealActive();

  // ---- mobile drawer: a modal dialog; focus moves in, stays in, and returns to the burger ----
  var burger = doc.querySelector(".burger");
  var backdrop = doc.getElementById("nav-backdrop");
  var tocClose = doc.querySelector(".toc-close");
  function tocFocusables() {
    return [].slice.call(toc.querySelectorAll("a[href], button")).filter(function (el) { return el.offsetParent !== null; });
  }
  function navOpen() {
    toc.classList.add("open"); backdrop.hidden = false; body.classList.add("nav-open");
    burger.setAttribute("aria-expanded", "true");
    toc.setAttribute("role", "dialog"); toc.setAttribute("aria-modal", "true"); toc.setAttribute("aria-label", "Navigation");
    revealActive();
    setTimeout(function () { (tocClose || toc).focus({ preventScroll: true }); revealActive(); }, 160);
  }
  function navClose(refocus) {
    toc.classList.remove("open"); backdrop.hidden = true; body.classList.remove("nav-open");
    burger.setAttribute("aria-expanded", "false");
    toc.removeAttribute("role"); toc.removeAttribute("aria-modal"); toc.removeAttribute("aria-label");
    if (refocus !== false) burger.focus();
  }
  if (burger) burger.addEventListener("click", function () { navOpen(); });
  if (backdrop) backdrop.addEventListener("click", function () { navClose(); });
  if (tocClose) tocClose.addEventListener("click", function () { navClose(); });
  doc.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && toc.classList.contains("open")) navClose();
  });
  if (toc) toc.addEventListener("keydown", function (e) {
    if (e.key !== "Tab" || !toc.classList.contains("open")) return;
    var f = tocFocusables(); if (!f.length) return;
    var first = f[0], last = f[f.length - 1];
    if (e.shiftKey && (doc.activeElement === first || doc.activeElement === toc)) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && doc.activeElement === last) { e.preventDefault(); first.focus(); }
  });

  // ---- arrow-key navigation: only when nothing else owns the arrow keys ----
  doc.addEventListener("keydown", function (e) {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return;
    var a = doc.activeElement;
    if (a && a !== body && a !== doc.documentElement && a.id !== "main") return;  // a focused control, table or code block keeps its arrows
    var ov = doc.getElementById("search-overlay"), lb = doc.getElementById("lightbox");
    if ((ov && !ov.hidden) || (lb && !lb.hidden) || (themeList && !themeList.hidden) || toc.classList.contains("open")) return;
    if (e.target && e.target.closest && e.target.closest("pre, .table-wrap")) return;
    var s = window.getSelection && window.getSelection();
    if (s && String(s).length) return;                                   // the reader is selecting text
    if (e.key === "ArrowRight" && body.dataset.next) { track("key_nav", { dir: "next" }); location.href = body.dataset.next; }
    if (e.key === "ArrowLeft" && body.dataset.prev) { track("key_nav", { dir: "prev" }); location.href = body.dataset.prev; }
  });

  // ---- search ----
  var overlay = doc.getElementById("search-overlay");
  var input = doc.getElementById("search-input");
  var results = doc.getElementById("search-results");
  var status = doc.getElementById("search-status");
  var searchClose = doc.getElementById("search-close");
  var index = null, sel = 0, lastFocus = null;

  function openSearch() {
    track("search_open");
    lastFocus = doc.activeElement;
    if (toc.classList.contains("open")) navClose(false);
    overlay.hidden = false;
    input.value = ""; results.innerHTML = ""; sel = 0;
    input.removeAttribute("aria-activedescendant");
    if (status) status.textContent = "";
    input.focus();
    if (!index) fetch("/search.json").then(function (r) { return r.json(); }).then(function (j) { index = j; });
  }
  function closeSearch() {
    overlay.hidden = true;
    // return focus to where the reader was, unless that was inside the (now closed) drawer
    var back = lastFocus && doc.contains(lastFocus) && !toc.contains(lastFocus) && lastFocus !== body
      ? lastFocus : (mq.matches ? burger : searchBtn);
    if (back && back.focus) back.focus();
  }

  if (searchBtn) searchBtn.addEventListener("click", openSearch);
  if (searchClose) searchClose.addEventListener("click", closeSearch);
  overlay.addEventListener("click", function (e) { if (e.target === overlay) closeSearch(); });
  doc.addEventListener("keydown", function (e) {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openSearch(); }
    if (e.key === "/" && overlay.hidden &&
        !(doc.activeElement && /INPUT|TEXTAREA/.test(doc.activeElement.tagName))) {
      e.preventDefault(); openSearch();
    }
    if (e.key === "Escape" && !overlay.hidden) closeSearch();
  });
  // Escape closes; Tab cycles input, results, close button
  overlay.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { e.preventDefault(); closeSearch(); return; }
    if (e.key !== "Tab") return;
    var f = [input].concat([].slice.call(results.querySelectorAll("a")), searchClose ? [searchClose] : []);
    var i = f.indexOf(doc.activeElement);
    e.preventDefault();
    f[(i + (e.shiftKey ? f.length - 1 : 1)) % f.length].focus();
  });
  // platform-appropriate shortcut label
  var kbd = doc.getElementById("search-kbd");
  if (kbd && !/Mac|iP/.test(navigator.platform)) kbd.textContent = "Ctrl K";

  function paintResults(items) {
    var shown = items.slice(0, 12);
    results.innerHTML = shown.map(function (it, i) {
      return '<li role="option" id="sr-' + i + '" aria-selected="' + (i === sel) + '"' + (i === sel ? ' class="sel"' : "") +
        '><a href="' + it.route + '" tabindex="-1">' +
        it.title + '<span class="sec">' + it.section + "</span>" +
        (it.text ? '<span class="snip">' + it.text + "</span>" : "") + "</a></li>";
    }).join("");
    if (shown.length) input.setAttribute("aria-activedescendant", "sr-" + sel);
    else input.removeAttribute("aria-activedescendant");
    if (status) status.textContent = !input.value.trim() ? "" :
      items.length ? items.length + (items.length === 1 ? " result" : " results") + (items.length > 12 ? ", showing the first 12" : "") : "No results";
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
    if (e.key === "Enter" && items[sel]) { track("search_go", { to: items[sel].route }); location.href = items[sel].route; }
  });

  // ---- diagrams: enlarge to near-native size in a full-window view ----
  var lightbox = doc.getElementById("lightbox");
  var lbBody = lightbox && lightbox.querySelector(".lightbox-body");
  var lbClose = lightbox && lightbox.querySelector(".lightbox-close");
  var lbFrom = null;
  function openLightbox(svg, from) {
    if (!lightbox) return;
    lbBody.innerHTML = "";
    var c = svg.cloneNode(true); c.classList.remove("diagram");
    lbBody.appendChild(c);
    lightbox.hidden = false; body.classList.add("nav-open"); lbFrom = from;
    lbClose.focus();
    track("diagram_zoom");
  }
  function closeLightbox() {
    lightbox.hidden = true; body.classList.remove("nav-open"); lbBody.innerHTML = "";
    if (lbFrom && lbFrom.focus) lbFrom.focus();
  }
  if (lightbox) {
    lbClose.addEventListener("click", closeLightbox);
    lightbox.addEventListener("keydown", function (e) {
      if (e.key === "Escape") { e.preventDefault(); closeLightbox(); }
      if (e.key === "Tab") { e.preventDefault(); lbClose.focus(); }
    });
    doc.querySelectorAll(".content svg.diagram").forEach(function (svg) {
      var p = doc.createElement("p"); p.className = "diagram-hint";
      var b = doc.createElement("button"); b.type = "button"; b.className = "diagram-zoom"; b.textContent = "Enlarge diagram";
      p.appendChild(b);
      svg.parentNode.insertBefore(p, svg.nextSibling);
      b.addEventListener("click", function () { openLightbox(svg, b); });
      svg.addEventListener("click", function () { openLightbox(svg, b); });
    });
  }
  // ---- outbound + pager click events ----
  doc.addEventListener("click", function (e) {
    var a = e.target.closest && e.target.closest("a");
    if (!a) return;
    if (a.classList.contains("pager")) track("pager", { dir: a.classList.contains("next") ? "next" : "prev" });
    else if (results.contains(a)) track("search_go", { to: a.getAttribute("href") });
    else if (a.target === "_blank") track("outbound", { url: (a.href || "").slice(0, 100) });
  });
})();

// ---- landing typing animation (original) ----
(function () {
  var holder = document.querySelector(".pixel-title");
  if (!holder || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  var svgs = [].slice.call(holder.querySelectorAll(".px-art"));
  var tag = holder.querySelector(".tagline");
  var tagText = tag ? tag.textContent : "";
  holder.classList.add("typing");
  if (tag) tag.innerHTML = '<span class="tag-text"></span><span class="tcur blink"></span>';

  function typeLine(svg, done) {
    if (!svg) return done();
    var letters = [].slice.call(svg.querySelectorAll(".px-letter"));
    if (!letters.length) return done();
    var cur = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    cur.setAttribute("class", "px-cursor blink");
    cur.setAttribute("y", "0");
    cur.setAttribute("width", "44");
    cur.setAttribute("height", "44");
    cur.setAttribute("x", letters[0].dataset.x);
    svg.appendChild(cur);
    var i = 0;
    setTimeout(function step() {
      letters[i].classList.add("on");
      var nxt = letters[i + 1];
      i++;
      if (nxt) {
        cur.setAttribute("x", nxt.dataset.x);
        setTimeout(step, 100 + Math.random() * 75);
      } else {
        cur.remove();
        setTimeout(done, 200);
      }
    }, 350);
  }

  typeLine(svgs[0], function () {
    typeLine(svgs[1], function () {
      if (!tag) return;
      var span = tag.querySelector(".tag-text"), i = 0;
      (function t() {
        span.textContent = tagText.slice(0, i);
        if (i++ <= tagText.length) setTimeout(t, 14 + Math.random() * 24);
      })();
    });
  });
})();
