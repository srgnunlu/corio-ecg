// Compact legal and product footer for the Corio ECG research prototype.

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="page-shell footer-grid">
        <div className="footer-brand">
          <a className="brand" href="#top"><span className="brand-mark"><i /></span><span className="brand-name">corio<span>/ECG</span></span></a>
          <p>Paper ECG intelligence,<br />built to be inspected.</p>
        </div>
        <div className="footer-links"><span>ÜRÜN</span><a href="#analiz">Analiz deneyimi</a><a href="#sistem">Nasıl çalışır</a><a href="#bilim">Bilimsel yaklaşım</a></div>
        <div className="footer-links"><span>PROJE</span><a href="#bilim">Araştırma kapsamı</a><a href="#top">Gizlilik yaklaşımı</a><a href="#top">Sorumluluk sınırı</a></div>
        <div className="footer-status"><span><i /> RESEARCH BUILD · 2026</span><p>Corio ECG klinik kullanıma onaylı bir tıbbi cihaz değildir.</p></div>
      </div>
      <div className="footer-bottom page-shell"><span>© 2026 Corio ECG</span><span>İstanbul · Türkiye</span><span>Made for clearer signals.</span></div>
    </footer>
  );
}
