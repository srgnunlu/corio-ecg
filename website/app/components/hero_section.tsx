// Cinematic first viewport introducing the paper-to-signal product promise.

import { EcgWave } from "./ecg_wave";

const METRICS = [
  ["12", "derivasyon"],
  ["150+", "model çıktısı"],
  ["500 Hz", "dijital sinyal"],
];

export function HeroSection() {
  return (
    <section className="hero" id="top" aria-labelledby="hero-title">
      <div className="hero-media" aria-hidden="true">
        <div className="hero-image" />
        <div className="hero-scan" />
        <div className="hero-noise" />
      </div>
      <div className="hero-content page-shell">
        <div className="hero-copy">
          <div className="eyebrow"><span /> AI-assisted ECG intelligence</div>
          <h1 id="hero-title">
            Kağıttaki sinyali<br />
            <em>klinik içgörüye</em> dönüştürün.
          </h1>
          <p>
            Corio, kağıt EKG fotoğrafını 12 derivasyonlu dijital sinyale çevirir;
            ölçümleri, 150’den fazla model çıktısını ve açıklanabilir raporu tek akışta sunar.
          </p>
          <div className="hero-actions">
            <a className="button button-primary" href="#analiz">
              EKG’nizi analiz edin <span>↗</span>
            </a>
            <a className="button button-ghost" href="#sistem">
              <span className="play-icon" aria-hidden="true">▶</span> Akışı izle
            </a>
          </div>
          <div className="hero-assurance">
            <span className="assurance-dot" /> Araştırma amaçlı karar destek prototipi
          </div>
        </div>
        <div className="hero-signal-card" aria-hidden="true">
          <div className="signal-card-top">
            <span>Live reconstruction</span><i>500 Hz</i>
          </div>
          <EcgWave compact />
          <div className="signal-card-bottom">
            <span>Lead II</span><span>25 mm/s</span><span>10 mm/mV</span>
          </div>
        </div>
      </div>
      <div className="hero-metrics page-shell" aria-label="Corio sistem özeti">
        {METRICS.map(([value, label]) => (
          <div className="hero-metric" key={label}>
            <strong>{value}</strong><span>{label}</span>
          </div>
        ))}
        <div className="hero-scroll"><i /> Keşfetmek için kaydırın</div>
      </div>
    </section>
  );
}
