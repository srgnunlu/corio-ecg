# Corio ECG Web

Premium product site and live upload experience for the Corio paper ECG
interpretation research pipeline.

## Local development

```bash
npm install
npm run dev
```

The frontend connects to the existing Gradio analysis service at
`http://127.0.0.1:7860` by default. Point it at another deployment when needed:

```bash
NEXT_PUBLIC_CORIO_API_URL=https://your-analysis-service.example npm run dev
```

## Product behavior

- accepts a paper ECG photo from file upload or mobile camera;
- calls Corio's `/analyze_ecg` Gradio endpoint;
- presents rhythm, intervals, image quality, model outputs, and reconstructed signal;
- downloads the PDF created by the same analysis session;
- offers an explicitly labeled sample report if the model service is unavailable.

## Validation

```bash
npm test
```

The production build targets Sites through vinext. Corio ECG remains a research
prototype and is not a clinically approved medical device.
