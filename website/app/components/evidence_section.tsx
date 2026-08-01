// Transparent internal benchmark evidence and clinical-boundary communication.

import { Reveal } from "./reveal";

const EVIDENCE = [
  { value: "97.73%", label: "Eşik uyumu", note: "eşleştirilmiş referans benchmark" },
  { value: "0.924", label: "Ortalama cosine tutarlılığı", note: "sinyal → fotoğraf → sinyal" },
  { value: "21,799", label: "PTB-XL kaydı", note: "araştırma değerlendirme tabanı" },
];

export function EvidenceSection() {
  return (
    <section className="evidence-section" id="bilim" aria-labelledby="evidence-title">
      <div className="page-shell">
        <Reveal className="evidence-heading">
          <span className="section-index">04 / BİLİM</span>
          <h2 id="evidence-title">Sihir değil.<br /><em>Ölçülebilir mühendislik.</em></h2>
          <p>Corio, yalnızca sonuç üretmek için değil; görüntüden sinyale kadar nerede hata oluşabileceğini göstermek için tasarlandı.</p>
        </Reveal>
        <div className="evidence-grid">
          {EVIDENCE.map((item, index) => (
            <Reveal className="evidence-metric" delay={index * 80} key={item.label}>
              <span>0{index + 1}</span><strong>{item.value}</strong><h3>{item.label}</h3><p>{item.note}</p>
            </Reveal>
          ))}
        </div>
        <Reveal className="evidence-note">
          <div><i>!</i><span><strong>Doğru bağlam önemlidir.</strong> Bu metrikler dahili araştırma benchmark’larıdır; klinik tanı doğruluğu veya tıbbi cihaz onayı anlamına gelmez.</span></div>
          <a href="#analiz">Araştırma prototipini deneyin ↗</a>
        </Reveal>
      </div>
    </section>
  );
}
