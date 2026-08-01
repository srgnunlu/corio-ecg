// Autoplaying product-film sequence showing the four stages of the Corio pipeline.

"use client";

import { useEffect, useState } from "react";
import { EcgWave } from "./ecg_wave";
import { Reveal } from "./reveal";

const STEPS = [
  {
    number: "01",
    title: "Fotoğrafı anlar",
    copy: "Perspektif, gölge ve kağıt düzeni analiz edilir; 12 derivasyonun konumu belirlenir.",
  },
  {
    number: "02",
    title: "Sinyali geri kazanır",
    copy: "Kağıttaki izler kalibre edilmiş, 500 Hz dijital EKG sinyaline dönüştürülür.",
  },
  {
    number: "03",
    title: "Bulguları uzlaştırır",
    copy: "150’den fazla model çıktısı, ritim ve aralık ölçümleriyle birlikte değerlendirilir.",
  },
  {
    number: "04",
    title: "Raporu anlatır",
    copy: "Teknik bulgular, hekim için ayrıntılı; hasta için anlaşılır bir rapora dönüşür.",
  },
];

function StepVisuals({ active }: { active: number }) {
  return (
    <div className="film-stage" aria-live="polite">
      <div className={`film-visual visual-photo ${active === 0 ? "is-active" : ""}`}>
        <div className="film-photo"><i /><span>Perspective lock</span></div>
        <div className="corner top-left" /><div className="corner top-right" />
        <div className="corner bottom-left" /><div className="corner bottom-right" />
      </div>
      <div className={`film-visual visual-signal ${active === 1 ? "is-active" : ""}`}>
        <div className="film-chip"><i /> SIGNAL RECONSTRUCTION</div>
        <EcgWave />
        <div className="lead-labels"><span>I</span><span>II</span><span>V1</span></div>
      </div>
      <div className={`film-visual visual-model ${active === 2 ? "is-active" : ""}`}>
        <span className="film-chip"><i /> ECGFOUNDER · 150+ OUTPUTS</span>
        {["SINUS RHYTHM", "NORMAL ECG", "ABNORMAL ECG", "ATRIAL FIBRILLATION"].map((label, index) => (
          <div className="model-row" key={label}>
            <span>{label}</span><div><i style={{ width: `${[91, 94, 8, 3][index]}%` }} /></div><strong>{[".91", ".94", ".08", ".03"][index]}</strong>
          </div>
        ))}
      </div>
      <div className={`film-visual visual-report ${active === 3 ? "is-active" : ""}`}>
        <div className="report-sheet">
          <div className="report-brand"><span className="brand-mark"><i /></span> corio/ECG <small>ANALİZ RAPORU</small></div>
          <div className="report-verdict"><i /> NORMAL EKG <strong>72 <small>bpm</small></strong></div>
          <p>Normal sinüs ritmi · Kritik öncelik görünmüyor</p>
          <EcgWave compact muted />
          <div className="report-measures"><span>PR <b>164</b></span><span>QRS <b>92</b></span><span>QTc <b>418</b></span></div>
        </div>
      </div>
      <span className="film-timecode">00:0{active + 1} / 00:04</span>
    </div>
  );
}

export function ProcessStory() {
  const [active, setActive] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => setActive((value) => (value + 1) % STEPS.length), 4200);
    return () => window.clearInterval(timer);
  }, [playing]);

  return (
    <section className="process-section" id="sistem" aria-labelledby="process-title">
      <div className="page-shell">
        <Reveal className="process-heading">
          <div><span className="section-index">02 / SİSTEM</span><h2 id="process-title">Bir fotoğraftan,<br /><em>izlenebilir bir karara.</em></h2></div>
          <p>Corio’nun her aşaması gözlemlenebilir. Görüntü kalitesi, sinyal rekonstrüksiyonu ve model bulguları raporda iz bırakır.</p>
        </Reveal>
        <Reveal className="process-film" delay={120}>
          <div className="film-label"><i>●</i> CORIO / PRODUCT FILM <button type="button" onClick={() => setPlaying((value) => !value)}>{playing ? "Durdur Ⅱ" : "Oynat ▶"}</button></div>
          <StepVisuals active={active} />
          <div className="film-steps">
            {STEPS.map((step, index) => (
              <button className={active === index ? "is-active" : ""} type="button" key={step.number} onClick={() => { setActive(index); setPlaying(false); }}>
                <i><span /></i><small>{step.number}</small><strong>{step.title}</strong><p>{step.copy}</p>
              </button>
            ))}
          </div>
        </Reveal>
      </div>
    </section>
  );
}
