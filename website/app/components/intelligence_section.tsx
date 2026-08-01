// Product capability narrative for patient communication, clinical depth, and traceability.

import { EcgWave } from "./ecg_wave";
import { Reveal } from "./reveal";

const PIPELINE = [
  ["01", "GÖRÜNTÜ", "Perspektif + gölge"],
  ["02", "SİNYAL", "12 derivasyon · 500 Hz"],
  ["03", "ÖLÇÜM", "HR · PR · QRS · QTc"],
  ["04", "MODEL", "150+ çıktı"],
  ["05", "RAPOR", "Yapılandırılmış sonuç"],
];

export function IntelligenceSection() {
  return (
    <section className="intelligence-section" aria-labelledby="intelligence-title">
      <div className="page-shell">
        <Reveal className="section-intro intelligence-intro">
          <span className="section-index">03 / ZEKÂ</span>
          <h2 id="intelligence-title">Sadece bir skor değil.<br /><em>Bağlamı olan bir sonuç.</em></h2>
        </Reveal>

        <div className="intelligence-grid">
          <Reveal className="intelligence-card patient-card" delay={70}>
            <span className="card-number">01</span><small>HASTA DİLİ</small>
            <h3>Teknik bulguyu,<br />anlaşılır bir sonraki adıma çevirir.</h3>
            <div className="conversation-card">
              <span>SONUÇ ÖZETİ</span>
              <p>Kalp ritminiz düzenli görünüyor. Ölçümler beklenen aralıkta; belirgin bir kritik öncelik görünmüyor.</p>
              <i>Sonuç hekiminiz tarafından doğrulanmalıdır.</i>
            </div>
          </Reveal>
          <Reveal className="intelligence-card clinician-card" delay={140}>
            <span className="card-number">02</span><small>KLİNİK DERİNLİK</small>
            <h3>Ham sinyale kadar<br />geri izlenebilir.</h3>
            <div className="clinical-monitor">
              <div><span>Lead II</span><i>GOOD SIGNAL</i></div>
              <EcgWave compact muted />
              <footer><span>PR 164 ms</span><span>QRS 92 ms</span><span>QTc 418 ms</span></footer>
            </div>
          </Reveal>
        </div>

        <Reveal className="pipeline-ribbon" delay={120}>
          <div className="pipeline-title"><span>UÇTAN UCA PIPELINE</span><i>Her adım denetlenebilir</i></div>
          <div className="pipeline-flow">
            {PIPELINE.map(([number, title, copy]) => (
              <div className="pipeline-step" key={number}>
                <small>{number}</small><strong>{title}</strong><span>{copy}</span>
              </div>
            ))}
          </div>
        </Reveal>
      </div>
    </section>
  );
}
