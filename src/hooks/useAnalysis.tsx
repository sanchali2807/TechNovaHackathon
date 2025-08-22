import React, { createContext, useContext, useState } from "react";

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectionFeatures {
  billboard_count: number;
  estimated_area_sqft?: number | null;
  bounding_boxes: BoundingBox[];
  qr_or_license_present?: boolean | null;
  text_content?: string[] | null;
}

export interface ComplianceCheck {
  id: string;
  type: "size" | "placement" | "content" | "license";
  name: string;
  status: "violation" | "compliant";
  passed: boolean;
  details: string;
  policy_reference?: string | null;
}

export interface ComplianceReport {
  overall_passed: boolean;
  checks: ComplianceCheck[];
}

export interface AnalysisResponse {
  file_id: string;
  filename: string;
  storage_url?: string | null;
  media_type: "image" | "video";
  detections: DetectionFeatures;
  compliance: ComplianceReport;
}

type AnalysisContextShape = {
  analysis: AnalysisResponse | null;
  previewDataUrl: string | null;
  setAnalysis: (a: AnalysisResponse | null, previewDataUrl?: string | null) => void;
};

const AnalysisContext = createContext<AnalysisContextShape | undefined>(undefined);

export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [analysis, setAnalysisState] = useState<AnalysisResponse | null>(null);
  const [previewDataUrl, setPreviewDataUrl] = useState<string | null>(null);

  const setAnalysis = (a: AnalysisResponse | null, preview?: string | null) => {
    setAnalysisState(a);
    if (typeof preview !== "undefined") setPreviewDataUrl(preview);
  };

  return (
    <AnalysisContext.Provider value={{ analysis, previewDataUrl, setAnalysis }}>
      {children}
    </AnalysisContext.Provider>
  );
};

export const useAnalysis = () => {
  const ctx = useContext(AnalysisContext);
  if (!ctx) throw new Error("useAnalysis must be used within AnalysisProvider");
  return ctx;
};



