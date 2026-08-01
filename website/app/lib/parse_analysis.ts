// Converts the trusted Gradio HTML response into the polished site result model.

import type {
  AnalysisTone,
  CaptureQuality,
  DiagnosisMetric,
  EcgAnalysisResult,
  GradioFileData,
  IntervalMetric,
} from "./analysis_types";

const ASSESSMENT_LABELS: Record<string, string> = {
  "Normal ECG": "Normal EKG",
  "Abnormal ECG": "Anormal EKG",
  "Indeterminate ECG": "Belirsiz EKG",
};

const RHYTHM_LABELS: Record<string, string> = {
  "Normal sinus rhythm": "Normal sinüs ritmi",
  "Sinus rhythm": "Sinüs ritmi",
  "Sinus bradycardia": "Sinüs bradikardisi",
  "Sinus tachycardia": "Sinüs taşikardisi",
  "Rhythm indeterminate": "Ritim belirsiz",
};

function documentFrom(html: string): Document {
  return new DOMParser().parseFromString(html, "text/html");
}

function cleanText(value: string | null | undefined): string {
  return (value ?? "").replace(/\s+/g, " ").trim();
}

function parseAssessment(text: string): { label: string; tone: AnalysisTone } {
  const source = Object.keys(ASSESSMENT_LABELS).find((label) => text.includes(label));
  if (source === "Normal ECG") return { label: ASSESSMENT_LABELS[source], tone: "normal" };
  if (source === "Abnormal ECG") return { label: ASSESSMENT_LABELS[source], tone: "abnormal" };
  return {
    label: source ? ASSESSMENT_LABELS[source] : "Belirsiz EKG",
    tone: "indeterminate",
  };
}

function parseRhythm(document: Document): { rhythm: string; heartRate: string } {
  const line = Array.from(document.querySelectorAll("div")).find((element) => {
    const text = cleanText(element.textContent);
    return text.startsWith("Rhythm:") && text.includes("HR:");
  });
  const text = cleanText(line?.textContent);
  const match = text.match(/Rhythm:\s*(.*?)\s*·\s*HR:\s*([\d.]+\s*bpm|n\/a)/i);
  const sourceRhythm = cleanText(match?.[1]) || "Ritim belirlenemedi";
  return {
    rhythm: RHYTHM_LABELS[sourceRhythm] ?? sourceRhythm,
    heartRate: match?.[2] ?? "—",
  };
}

function parseIntervals(document: Document): IntervalMetric[] {
  return Array.from(document.querySelectorAll("td"))
    .map((cell) => {
      const text = cleanText(cell.textContent);
      const label = ["QTc", "QRS", "PR", "QT"].find((candidate) =>
        text.startsWith(candidate),
      );
      const value = text.match(/([\d.]+\s*ms)/i)?.[1];
      if (!label || !value) return null;
      const note = cleanText(text.replace(label, "").replace(value, ""));
      return { label, value, note: note || undefined };
    })
    .filter((metric): metric is IntervalMetric => metric !== null)
    .slice(0, 4);
}

function parseDiagnoses(html: string): DiagnosisMetric[] {
  const document = documentFrom(html);
  return Array.from(document.querySelectorAll("tbody tr"))
    .map((row) => {
      const cells = row.querySelectorAll("td");
      const label = cleanText(cells[0]?.textContent);
      const probability = Number.parseFloat(cleanText(cells[2]?.textContent));
      return label && Number.isFinite(probability) ? { label, probability } : null;
    })
    .filter((metric): metric is DiagnosisMetric => metric !== null)
    .slice(0, 6);
}

function parseQuality(html: string): CaptureQuality {
  const document = documentFrom(html);
  const items = Array.from(document.querySelectorAll("div")).map((element) =>
    cleanText(element.textContent),
  );
  const findValue = (prefix: string) =>
    items.find((item) => item.startsWith(prefix) && item.length < 80)?.slice(prefix.length).trim();
  return {
    grade: findValue("Quality:") ?? "—",
    layout: findValue("Layout:") ?? "—",
    detectedLeads: findValue("Detected leads:") ?? "—",
  };
}

export function parseAnalysisResponse(data: unknown[]): EcgAnalysisResult {
  const waveform = data[0] as GradioFileData | null;
  const criticalHtml = typeof data[1] === "string" ? data[1] : "";
  const diagnosesHtml = typeof data[2] === "string" ? data[2] : "";
  const debugHtml = typeof data[3] === "string" ? data[3] : "";
  const document = documentFrom(criticalHtml);
  const text = cleanText(document.body.textContent);
  const assessment = parseAssessment(text);
  const rhythm = parseRhythm(document);
  const reasons = Array.from(document.querySelectorAll("ul:first-of-type li"))
    .map((item) => cleanText(item.textContent))
    .filter(Boolean)
    .slice(0, 4);
  const narrative = Array.from(document.querySelectorAll("p"))
    .map((paragraph) => cleanText(paragraph.textContent))
    .filter(Boolean)
    .slice(0, 3);

  return {
    assessment: assessment.label,
    tone: assessment.tone,
    rhythm: rhythm.rhythm,
    heartRate: rhythm.heartRate,
    reasons,
    narrative,
    intervals: parseIntervals(document),
    diagnoses: parseDiagnoses(diagnosesHtml),
    quality: parseQuality(debugHtml),
    waveformUrl: waveform?.url ?? null,
    criticalHtml,
    diagnosesHtml,
    debugHtml,
    isSample: false,
  };
}

export const SAMPLE_RESULT: EcgAnalysisResult = {
  assessment: "Normal EKG",
  tone: "normal",
  rhythm: "Normal sinüs ritmi",
  heartRate: "72 bpm",
  reasons: ["Belirgin kritik bulgu saptanmadı."],
  narrative: [
    "Örnek kayıt düzenli sinüs ritmi ile uyumludur. Ölçülen aralıklar referans sınırlar içindedir.",
    "Bu ekran yalnızca arayüz örneğidir; yüklediğiniz görüntünün tıbbi analizi değildir.",
  ],
  intervals: [
    { label: "PR", value: "164 ms", note: "referans içinde" },
    { label: "QRS", value: "92 ms", note: "dar kompleks" },
    { label: "QT", value: "388 ms" },
    { label: "QTc", value: "418 ms", note: "Framingham" },
  ],
  diagnoses: [
    { label: "NORMAL ECG", probability: 0.94 },
    { label: "SINUS RHYTHM", probability: 0.91 },
    { label: "ABNORMAL ECG", probability: 0.08 },
  ],
  quality: { grade: "GOOD", layout: "3x4+1R", detectedLeads: "12/12" },
  waveformUrl: null,
  criticalHtml: "",
  diagnosesHtml: "",
  debugHtml: "",
  isSample: true,
};
