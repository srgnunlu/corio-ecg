// Corio ECG premium landing page and live paper-ECG analysis experience.

import { AnalysisStudio } from "./components/analysis_studio";
import { EvidenceSection } from "./components/evidence_section";
import { FinalCta } from "./components/final_cta";
import { HeroSection } from "./components/hero_section";
import { IntelligenceSection } from "./components/intelligence_section";
import { ProcessStory } from "./components/process_story";
import { SiteFooter } from "./components/site_footer";
import { SiteHeader } from "./components/site_header";

export default function Home() {
  return (
    <>
      <SiteHeader />
      <main>
        <HeroSection />
        <AnalysisStudio />
        <ProcessStory />
        <IntelligenceSection />
        <EvidenceSection />
        <FinalCta />
      </main>
      <SiteFooter />
    </>
  );
}
