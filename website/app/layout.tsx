// Root document metadata, fonts, and social preview configuration for Corio ECG.

import type { Metadata } from "next";
import { headers } from "next/headers";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.includes("localhost") ? "http" : "https");
  const baseUrl = `${protocol}://${host}`;
  const title = "Corio ECG — Kağıt EKG’den Klinik İçgörüye";
  const description = "Kağıt EKG fotoğrafını dijital sinyale, ölçümlere, AI bulgularına ve açıklanabilir rapora dönüştüren araştırma platformu.";

  return {
    metadataBase: new URL(baseUrl),
    title,
    description,
    applicationName: "Corio ECG",
    keywords: ["EKG", "ECG", "yapay zekâ", "kağıt EKG", "sinyal dijitalleştirme"],
    icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
    openGraph: {
      title,
      description,
      type: "website",
      locale: "tr_TR",
      images: [{ url: `${baseUrl}/og.png`, width: 1734, height: 907, alt: "Corio ECG — Kağıttaki sinyali klinik içgörüye dönüştürün" }],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [`${baseUrl}/og.png`],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="tr" className="scroll-smooth">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body>
    </html>
  );
}
