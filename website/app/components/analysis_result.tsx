// High-trust patient summary and clinician detail presentation for one ECG analysis.

"use client";

import type { EcgAnalysisResult } from "../lib/analysis_types";
import { EcgWave } from "./ecg_wave";

type AnalysisResultProps = {
  result: EcgAnalysisResult;
  onDownload: () => void;
  downloading: boolean;
};

const TONE_COPY = {
  normal: "Kritik öncelik görünmüyor",
  abnormal: "Hekim değerlendirmesi gerekli",
  indeterminate: "Ek değerlendirme gerekli",
};

function PatientSummary({ result }: { result: EcgAnalysisResult }) {
  const fallback = result.tone === "normal"
    ? "Analiz, belirgin kritik bir bulgu göstermedi. Yine de sonuç bir hekim tarafından doğrulanmalıdır."
    : "Analizde doğrulanması gereken bulgular görüldü. Sonuç, EKG kaydıyla birlikte bir hekim tarafından değerlendirilmelidir.";
  return (
    <div className="patient-summary">
      <span>Hasta için anlaşılır özet</span>
      <p>{result.narrative[0] || fallback}</p>
      {result.isSample && (
        <div className="sample-warning">Bu, yüklediğiniz görüntünün analizi değil; temsili rapor önizlemesidir.</div>
      )}
    </div>
  );
}

export function AnalysisResult({ result, onDownload, downloading }: AnalysisResultProps) {
  return (
    <div className={`analysis-result tone-${result.tone}`} aria-live="polite">
      <div className="result-head">
        <div>
          <span className="result-kicker">
            <i /> {result.isSample ? "Temsili rapor" : "Canlı model çıktısı"}
          </span>
          <h3>{result.assessment}</h3>
          <p>{TONE_COPY[result.tone]}</p>
        </div>
        <div className="result-rate"><strong>{result.heartRate.replace(" bpm", "")}</strong><span>bpm</span></div>
      </div>

      <PatientSummary result={result} />

      <div className="result-rhythm-row">
        <div><span>Ritim</span><strong>{result.rhythm}</strong></div>
        <div><span>Görüntü kalitesi</span><strong>{result.quality.grade}</strong></div>
        <div><span>Derivasyon</span><strong>{result.quality.detectedLeads}</strong></div>
      </div>

      <div className="result-wave">
        <div className="result-section-label"><span>Dijitalize sinyal</span><i>25 mm/s · 10 mm/mV</i></div>
        {result.waveformUrl ? (
          // The URL is returned by the trusted, configured Corio analysis service.
          // eslint-disable-next-line @next/next/no-img-element
          <img src={result.waveformUrl} alt="Analizden üretilen 12 derivasyonlu EKG sinyali" />
        ) : (
          <EcgWave muted />
        )}
      </div>

      <div className="interval-grid" aria-label="EKG aralık ölçümleri">
        {(result.intervals.length ? result.intervals : [
          { label: "PR", value: "—" },
          { label: "QRS", value: "—" },
          { label: "QT", value: "—" },
          { label: "QTc", value: "—" },
        ]).map((metric, index) => (
          <div className="interval-card" key={`${metric.label}-${index}`}>
            <span>{metric.label}</span><strong>{metric.value}</strong><small>{metric.note || "ölçüm"}</small>
          </div>
        ))}
      </div>

      <div className="diagnosis-panel">
        <div className="result-section-label"><span>En güçlü model çıktıları</span><i>eşik ≥ 0.70</i></div>
        {result.diagnoses.slice(0, 4).map((diagnosis) => (
          <div className="diagnosis-row" key={diagnosis.label}>
            <span>{diagnosis.label}</span>
            <div><i style={{ width: `${Math.max(3, diagnosis.probability * 100)}%` }} /></div>
            <strong>{diagnosis.probability.toFixed(2)}</strong>
          </div>
        ))}
      </div>

      {result.reasons.length > 0 && (
        <div className="finding-list">
          <span>Değerlendirme notları</span>
          <ul>{result.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
        </div>
      )}

      <div className="result-actions">
        <button className="button button-primary" type="button" onClick={onDownload} disabled={downloading}>
          {downloading ? "Rapor hazırlanıyor…" : result.isSample ? "Raporu yazdır" : "PDF raporu indir"}
        </button>
        <span>Son karar, EKG’yi inceleyen yetkin sağlık profesyoneline aittir.</span>
      </div>

      {!result.isSample && result.criticalHtml && (
        <details className="raw-report">
          <summary>Teknik rapor ayrıntılarını aç</summary>
          <div className="backend-html" dangerouslySetInnerHTML={{ __html: result.criticalHtml }} />
        </details>
      )}
    </div>
  );
}
