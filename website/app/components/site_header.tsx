// Persistent site navigation and Corio brand mark.

"use client";

import { useEffect, useState } from "react";

const NAV_ITEMS = [
  { label: "Deneyim", href: "#analiz" },
  { label: "Sistem", href: "#sistem" },
  { label: "Bilim", href: "#bilim" },
];

export function SiteHeader() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header className={`site-header ${scrolled ? "is-scrolled" : ""}`}>
      <a className="brand" href="#top" aria-label="Corio ECG ana sayfa">
        <span className="brand-mark" aria-hidden="true"><i /></span>
        <span className="brand-name">corio<span>/ECG</span></span>
      </a>
      <nav className={`site-nav ${open ? "is-open" : ""}`} aria-label="Ana menü">
        {NAV_ITEMS.map((item) => (
          <a key={item.href} href={item.href} onClick={() => setOpen(false)}>
            {item.label}
          </a>
        ))}
      </nav>
      <a className="header-cta" href="#analiz">Bir EKG yükle <span>↗</span></a>
      <button
        className="menu-button"
        type="button"
        aria-label={open ? "Menüyü kapat" : "Menüyü aç"}
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span /><span />
      </button>
    </header>
  );
}
