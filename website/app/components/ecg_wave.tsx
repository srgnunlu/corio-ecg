// Animated canvas waveform used across the Corio storytelling and result views.

"use client";

import { useEffect, useRef } from "react";

type EcgWaveProps = {
  className?: string;
  compact?: boolean;
  muted?: boolean;
};

function gaussian(value: number, center: number, width: number): number {
  const distance = (value - center) / width;
  return Math.exp(-distance * distance);
}

function ecgSample(position: number): number {
  const cycle = ((position % 1) + 1) % 1;
  return (
    0.08 * gaussian(cycle, 0.18, 0.05) -
    0.12 * gaussian(cycle, 0.355, 0.014) +
    1.05 * gaussian(cycle, 0.39, 0.012) -
    0.24 * gaussian(cycle, 0.425, 0.018) +
    0.26 * gaussian(cycle, 0.68, 0.085)
  );
}

export function EcgWave({ className = "", compact = false, muted = false }: EcgWaveProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const context = canvas.getContext("2d");
    if (!context) return undefined;

    let animationFrame = 0;
    let phase = 0;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const draw = () => {
      const bounds = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.floor(bounds.width * ratio));
      const height = Math.max(1, Math.floor(bounds.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, bounds.width, bounds.height);

      const gridSize = compact ? 12 : 18;
      context.lineWidth = 0.55;
      context.strokeStyle = muted ? "rgba(83, 166, 153, .08)" : "rgba(75, 238, 208, .10)";
      for (let x = 0; x < bounds.width; x += gridSize) {
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, bounds.height);
        context.stroke();
      }
      for (let y = 0; y < bounds.height; y += gridSize) {
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(bounds.width, y);
        context.stroke();
      }

      const leadCount = compact ? 1 : 3;
      for (let lead = 0; lead < leadCount; lead += 1) {
        const baseline = ((lead + 1) * bounds.height) / (leadCount + 1);
        const amplitude = compact ? bounds.height * 0.23 : bounds.height * 0.105;
        context.beginPath();
        for (let x = 0; x <= bounds.width; x += 1.5) {
          const samplePosition = x / (compact ? 122 : 150) + phase + lead * 0.08;
          const noise = Math.sin(x * 0.041 + lead) * 0.008;
          const y = baseline - (ecgSample(samplePosition) + noise) * amplitude;
          if (x === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        }
        context.lineWidth = compact ? 1.7 : 1.35;
        context.strokeStyle = muted ? "rgba(110, 190, 176, .66)" : "rgba(42, 238, 204, .92)";
        context.shadowColor = "rgba(38, 232, 199, .42)";
        context.shadowBlur = muted ? 0 : 7;
        context.stroke();
        context.shadowBlur = 0;
      }

      if (!reducedMotion) {
        phase += 0.0028;
        animationFrame = window.requestAnimationFrame(draw);
      }
    };

    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    draw();
    return () => {
      observer.disconnect();
      window.cancelAnimationFrame(animationFrame);
    };
  }, [compact, muted]);

  return (
    <canvas
      ref={canvasRef}
      className={`ecg-canvas ${className}`}
      role="img"
      aria-label="Animasyonlu temsili EKG sinyali"
    />
  );
}
