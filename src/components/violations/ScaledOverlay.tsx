import React, { useEffect, useRef, useState } from "react";

type Box = { x: number; y: number; width: number; height: number };

export const ScaledOverlay: React.FC<{ preview: string; boxes: Box[] }> = ({ preview, boxes }) => {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [scale, setScale] = useState({ sx: 1, sy: 1, ox: 0, oy: 0 });

  useEffect(() => {
    const compute = () => {
      const img = imgRef.current;
      const container = containerRef.current;
      if (!img || !container) return;
      const naturalW = img.naturalWidth || 1;
      const naturalH = img.naturalHeight || 1;
      const viewW = container.clientWidth;
      const viewH = Math.min(360, Math.round((naturalH / naturalW) * viewW));
      const scaleFactor = Math.min(viewW / naturalW, viewH / naturalH);
      const renderW = naturalW * scaleFactor;
      const renderH = naturalH * scaleFactor;
      const offsetX = (viewW - renderW) / 2;
      const offsetY = (viewH - renderH) / 2;
      setScale({ sx: scaleFactor, sy: scaleFactor, ox: offsetX, oy: offsetY });
    };
    compute();
    window.addEventListener("resize", compute);
    return () => window.removeEventListener("resize", compute);
  }, [preview]);

  return (
    <div ref={containerRef} className="mt-4 relative rounded-md overflow-hidden border border-border" style={{ height: 360 }}>
      <img ref={imgRef} src={preview} className="absolute inset-0 w-full h-full object-contain bg-black/50" />
      {boxes.map((b, i) => (
        <div
          key={i}
          className="absolute border-2 border-violation-red"
          style={{
            left: Math.round(scale.ox + b.x * scale.sx),
            top: Math.round(scale.oy + b.y * scale.sy),
            width: Math.round(b.width * scale.sx),
            height: Math.round(b.height * scale.sy),
          }}
        />
      ))}
    </div>
  );
};



