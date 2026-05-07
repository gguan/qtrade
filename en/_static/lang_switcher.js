// Rewrites the language-switcher link in the announcement banner so it
// points at the same page in the *other* language. Path layout assumed:
//
//   <project base>/en/<page>   ←→   <project base>/zh/<page>
//
// Works regardless of how deep the docs are nested under a base path
// (e.g. `/qtrade/en/...` on GitHub Pages).
(function () {
    const link = document.getElementById("lang-switcher-link");
    if (!link) return;
    const path = window.location.pathname;
    let target, label;
    if (path.includes("/zh/")) {
        target = path.replace("/zh/", "/en/");
        label = "EN ↗";
    } else if (path.includes("/en/")) {
        target = path.replace("/en/", "/zh/");
        label = "中文 ↗";
    } else {
        // No /en/ or /zh/ prefix — site root or local single-language build.
        target = path.replace(/\/$/, "") + "/zh/";
        label = "中文 ↗";
    }
    link.href = target;
    link.textContent = label;
})();
