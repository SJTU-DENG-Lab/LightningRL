$(document).ready(function () {
  // Navbar burger toggle for mobile
  $(".navbar-burger").click(function () {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  // Language toggle
  var savedLang = localStorage.getItem("lightningrl-lang") || "en";
  setLanguage(savedLang);

  $(".lang-toggle").click(function () {
    var current = $("body").hasClass("lang-zh") ? "zh" : "en";
    var next = current === "en" ? "zh" : "en";
    setLanguage(next);
    localStorage.setItem("lightningrl-lang", next);
  });

  function setLanguage(lang) {
    if (lang === "zh") {
      $("body").removeClass("lang-en").addClass("lang-zh");
      $(".lang-toggle .lang-en-btn").removeClass("active");
      $(".lang-toggle .lang-zh-btn").addClass("active");
    } else {
      $("body").removeClass("lang-zh").addClass("lang-en");
      $(".lang-toggle .lang-en-btn").addClass("active");
      $(".lang-toggle .lang-zh-btn").removeClass("active");
    }
  }
});
