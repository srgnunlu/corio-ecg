// Interactive ECG upload studio connected to the existing Corio Gradio backend.

"use client";

import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { analyzeEcg, createPdfReport } from "../lib/analysis_client";
import type { AnalysisStage, EcgAnalysisResult } from "../lib/analysis_types";
import { SAMPLE_RESULT } from "../lib/parse_analysis";
import { AnalysisResult } from "./analysis_result";
import { Reveal } from "./reveal";

const STAGE_META: Record<AnalysisStage, { label: string; progress: number }> = {
  idle: { label: "Fotoğrafınızı bekliyor", progress: 0 },
  ready: { label: "Analize hazır", progress: 4 },
  connecting: { label: "Görüntü güvenli analiz akışına alınıyor", progress: 14 },
  digitizing: { label: "Kağıt izleri 500 Hz sinyale dönüştürülüyor", progress: 46 },
  interpreting: { label: "150+ model çıktısı değerlendiriliyor", progress: 72 },
  reporting: { label: "Açıklanabilir rapor hazırlanıyor", progress: 89 },
  complete: { label: "Analiz tamamlandı", progress: 100 },
  unavailable: { label: "Analiz motoruna ulaşılamadı", progress: 0 },
};

const LAYOUTS = ["Auto-detect", "3x4+1R (standard)", "3x4+3R", "6x2", "6x2+1R"];

export function AnalysisStudio() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [layout, setLayout] = useState("3x4+1R (standard)");
  const [stage, setStage] = useState<AnalysisStage>("idle");
  const [result, setResult] = useState<EcgAnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const timerIds = useRef<number[]>([]);

  useEffect(() => () => {
    timerIds.current.forEach(window.clearTimeout);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const selectFile = (candidate: File | undefined) => {
    if (!candidate) return;
    if (!candidate.type.startsWith("image/")) {
      setError("Lütfen JPG, PNG, HEIC veya WebP formatında bir görüntü seçin.");
      return;
    }
    if (candidate.size > 20 * 1024 * 1024) {
      setError("Görüntü 20 MB sınırını aşıyor. Daha küçük bir kopya yükleyin.");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(candidate);
    setPreviewUrl(URL.createObjectURL(candidate));
    setStage("ready");
    setResult(null);
    setError("");
  };

  const onInput = (event: ChangeEvent<HTMLInputElement>) => selectFile(event.target.files?.[0]);
  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  };

  const clearTimers = () => {
    timerIds.current.forEach(window.clearTimeout);
    timerIds.current = [];
  };

  const beginAnalysis = async () => {
    if (!file) return;
    clearTimers();
    setResult(null);
    setError("");
    setStage("connecting");
    timerIds.current = [
      window.setTimeout(() => setStage("digitizing"), 1400),
      window.setTimeout(() => setStage("interpreting"), 6500),
      window.setTimeout(() => setStage("reporting"), 15000),
    ];
    try {
      const analysis = await analyzeEcg(file, layout);
      clearTimers();
      setResult(analysis);
      setStage("complete");
    } catch {
      clearTimers();
      setStage("unavailable");
      setError("Canlı model servisi şu anda erişilebilir değil. Yerel Corio analiz motorunu açıp tekrar deneyebilir veya temsili raporu inceleyebilirsiniz.");
    }
  };

  const showSample = () => {
    clearTimers();
    setResult(SAMPLE_RESULT);
    setStage("complete");
    setError("");
  };

  const downloadReport = async () => {
    if (!result) return;
    if (result.isSample) {
      window.print();
      return;
    }
    setDownloading(true);
    try {
      const url = await createPdfReport();
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "corio-ecg-raporu.pdf";
      anchor.click();
    } catch {
      setError("PDF raporu hazırlanamadı. Sonucu ekranda incelemeye devam edebilirsiniz.");
    } finally {
      setDownloading(false);
    }
  };

  const stageMeta = STAGE_META[stage];
  const isRunning = ["connecting", "digitizing", "interpreting", "reporting"].includes(stage);

  return (
    <section className="analysis-section" id="analiz" aria-labelledby="analysis-title">
      <div className="page-shell">
        <Reveal className="section-intro analysis-intro">
          <span className="section-index">01 / DENEYİM</span>
          <h2 id="analysis-title">Fotoğrafı bırakın.<br /><em>Corio sinyali okusun.</em></h2>
          <p>Net, gölgesiz ve tüm kağıdın göründüğü bir EKG fotoğrafı en iyi sonucu verir. Hasta kimlik bilgilerini yüklemeden önce kapatın.</p>
        </Reveal>

        <div className="analysis-workspace">
          <Reveal className="upload-panel" delay={100}>
            <div className="panel-topline"><span>EKG GÖRSELİ</span><i>JPG · PNG · HEIC · WEBP</i></div>
            <label
              className={`drop-zone ${dragging ? "is-dragging" : ""} ${previewUrl ? "has-image" : ""}`}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <input type="file" accept="image/*" capture="environment" onChange={onInput} />
              {previewUrl ? (
                // User-selected object URL never leaves the page until analysis starts.
                // eslint-disable-next-line @next/next/no-img-element
                <img src={previewUrl} alt="Yüklenmek üzere seçilen EKG fotoğrafı" />
              ) : (
                <div className="drop-empty">
                  <span className="upload-orbit"><i>＋</i></span>
                  <strong>EKG fotoğrafını buraya bırakın</strong>
                  <p>veya cihazınızdan seçmek için dokunun</p>
                  <small>En fazla 20 MB · Kimlik bilgilerini kapatın</small>
                </div>
              )}
              {previewUrl && isRunning && <div className="image-scan"><i /></div>}
              {previewUrl && <span className="replace-image">Görüntüyü değiştir</span>}
            </label>

            <div className="analysis-controls">
              <label>Kağıt düzeni
                <select value={layout} onChange={(event) => setLayout(event.target.value)} disabled={isRunning}>
                  {LAYOUTS.map((option) => <option key={option}>{option}</option>)}
                </select>
              </label>
              <div className="privacy-chip"><i>✓</i><span>Arayüz kalıcı kopya tutmaz</span></div>
            </div>

            <button className="analyze-button" type="button" disabled={!file || isRunning} onClick={beginAnalysis}>
              <span>{isRunning ? "Analiz sürüyor" : "Analizi başlat"}</span><i>{isRunning ? "•••" : "↗"}</i>
            </button>

            <div className={`analysis-progress ${isRunning ? "is-running" : ""}`}>
              <div><span>{stageMeta.label}</span><strong>{stageMeta.progress}%</strong></div>
              <progress value={stageMeta.progress} max="100">{stageMeta.progress}%</progress>
            </div>

            {error && (
              <div className="analysis-error" role="alert">
                <p>{error}</p>
                {stage === "unavailable" && <button type="button" onClick={showSample}>Temsili raporu aç →</button>}
              </div>
            )}
          </Reveal>

          <Reveal className="result-panel" delay={180}>
            {result ? (
              <AnalysisResult result={result} onDownload={downloadReport} downloading={downloading} />
            ) : (
              <div className="result-placeholder">
                <span className="result-orbit"><i /><b>12</b></span>
                <div><small>SONUÇ ALANI</small><h3>Analiziniz burada<br />hayat bulacak.</h3></div>
                <ul><li>Ritim ve kalp hızı</li><li>PR · QRS · QT · QTc</li><li>Açıklanabilir bulgular</li><li>PDF raporu</li></ul>
                <button type="button" onClick={showSample}>Örnek sonucu incele <span>↗</span></button>
              </div>
            )}
          </Reveal>
        </div>
      </div>
    </section>
  );
}
