// Final conversion section that returns the visitor to the upload experience.

import { EcgWave } from "./ecg_wave";
import { Reveal } from "./reveal";

export function FinalCta() {
  return (
    <section className="final-cta" aria-labelledby="final-title">
      <div className="final-cta-media" aria-hidden="true"><EcgWave /></div>
      <Reveal className="final-cta-content page-shell">
        <span>Fotoğraftan içgörüye</span>
        <h2 id="final-title">Her çizgi bir hikâye taşır.<br /><em>Corio onu görünür kılar.</em></h2>
        <a className="button button-light" href="#analiz">İlk analizi başlat <span>↗</span></a>
        <p>Araştırma ve eğitim amaçlıdır · Sonuçlar hekim değerlendirmesi gerektirir</p>
      </Reveal>
    </section>
  );
}
