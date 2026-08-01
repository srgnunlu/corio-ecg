// Browser bridge between the premium frontend and the existing Gradio model service.

import { Client } from "@gradio/client";
import type { EcgAnalysisResult, GradioFileData } from "./analysis_types";
import { parseAnalysisResponse } from "./parse_analysis";

const DEFAULT_ENDPOINT = "http://127.0.0.1:7860";
let activeClient: Client | null = null;
let activeEndpoint = "";

function endpointUrl(): string {
  return process.env.NEXT_PUBLIC_CORIO_API_URL?.trim() || DEFAULT_ENDPOINT;
}

async function getClient(): Promise<Client> {
  const endpoint = endpointUrl();
  if (!activeClient || activeEndpoint !== endpoint) {
    activeClient?.close();
    activeClient = await Client.connect(endpoint);
    activeEndpoint = endpoint;
  }
  return activeClient;
}

export async function analyzeEcg(
  image: File,
  layoutChoice: string,
  threshold = 0.7,
): Promise<EcgAnalysisResult> {
  const client = await getClient();
  const response = await client.predict<unknown[]>("/analyze_ecg", {
    image,
    threshold,
    layout_choice: layoutChoice,
    language: "tr",
  });
  return parseAnalysisResponse(response.data);
}

export async function createPdfReport(): Promise<string> {
  const client = await getClient();
  const response = await client.predict<GradioFileData[]>("/generate_pdf", {});
  const url = response.data[0]?.url;
  if (!url) throw new Error("PDF report URL was not returned.");
  return url;
}

export function configuredEndpoint(): string {
  return endpointUrl();
}
