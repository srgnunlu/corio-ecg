// Shared types for the live Corio ECG analysis experience.

export type AnalysisStage =
  | "idle"
  | "ready"
  | "connecting"
  | "digitizing"
  | "interpreting"
  | "reporting"
  | "complete"
  | "unavailable";

export type AnalysisTone = "normal" | "abnormal" | "indeterminate";

export type IntervalMetric = {
  label: string;
  value: string;
  note?: string;
};

export type DiagnosisMetric = {
  label: string;
  probability: number;
};

export type CaptureQuality = {
  grade: string;
  layout: string;
  detectedLeads: string;
};

export type EcgAnalysisResult = {
  assessment: string;
  tone: AnalysisTone;
  rhythm: string;
  heartRate: string;
  reasons: string[];
  narrative: string[];
  intervals: IntervalMetric[];
  diagnoses: DiagnosisMetric[];
  quality: CaptureQuality;
  waveformUrl: string | null;
  criticalHtml: string;
  diagnosesHtml: string;
  debugHtml: string;
  isSample: boolean;
};

export type GradioFileData = {
  path?: string | null;
  url?: string | null;
  orig_name?: string | null;
};
