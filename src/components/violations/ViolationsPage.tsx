import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, CheckCircle, XCircle, Eye, FileText } from "lucide-react";
import { useAnalysis } from "@/hooks/useAnalysis";
import { ScaledOverlay } from "./ScaledOverlay";

interface ViolationsPageProps {
  onNavigate: (page: string) => void;
}

export const ViolationsPage = ({ onNavigate }: ViolationsPageProps) => {
  const { analysis, previewDataUrl } = useAnalysis();
  const checks = analysis?.compliance.checks || [];
  const noBillboard = (analysis?.detections?.billboard_count || 0) === 0;
  const violationCount = noBillboard ? 0 : checks.filter(c => c.status === "violation").length;
  const complianceCount = noBillboard ? 0 : checks.filter(c => c.status === "compliant").length;

  return (
    <div className="min-h-screen bg-background pb-20">
      {/* Header */}
      <div className="bg-card border-b border-border px-6 py-4 flex items-center">
        <Button 
          variant="ghost" 
          size="sm" 
          onClick={() => onNavigate("camera")}
          className="mr-3 p-2"
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-xl font-semibold text-foreground">Detection Results</h1>
      </div>

      <div className="px-6 py-6 space-y-6">
        {/* Summary Cards */}
        <div className="grid grid-cols-2 gap-4 animate-slide-up">
          <Card className="p-4 shadow-card border-violation-red bg-violation-red/5">
            <div className="text-center space-y-2">
              <XCircle className="h-8 w-8 text-violation-red mx-auto" />
              <div className="text-2xl font-bold text-violation-red">{violationCount}</div>
              <div className="text-sm text-muted-foreground">Violations</div>
            </div>
          </Card>
          
          <Card className="p-4 shadow-card border-compliance-green bg-compliance-green/5">
            <div className="text-center space-y-2">
              <CheckCircle className="h-8 w-8 text-compliance-green mx-auto" />
              <div className="text-2xl font-bold text-compliance-green">{complianceCount}</div>
              <div className="text-sm text-muted-foreground">Compliant</div>
            </div>
          </Card>
        </div>

        {/* AI Analysis Header with preview and boxes */}
        <Card className="p-4 gradient-card shadow-elevated animate-slide-in">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-foreground">AI Analysis Complete</h2>
              <p className="text-sm text-muted-foreground">Scanned on {new Date().toLocaleString()}</p>
            </div>
            <Eye className="h-8 w-8 text-primary" />
          </div>
          {noBillboard ? (
            <div className="mt-4 p-4 border border-border rounded-md bg-muted/30 text-sm text-muted-foreground">
              No billboard detected in this image. Please ensure the frame clearly contains the full billboard and try again.
            </div>
          ) : previewDataUrl && analysis?.detections?.bounding_boxes?.length ? (
            <ScaledOverlay preview={previewDataUrl} boxes={analysis.detections.bounding_boxes} />
          ) : null}
        </Card>

        {/* Violations List */}
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-foreground">Detection Details</h3>
          
          {!noBillboard && checks.map((c, index) => {
            const isViolation = c.status === "violation";
            return (
              <Card
                key={`${c.id}-${index}`}
                className={`p-4 shadow-card animate-slide-in border-l-4 ${isViolation ? "border-l-violation-red bg-violation-red/5" : "border-l-compliance-green bg-compliance-green/5"}`}
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="flex items-start space-x-3">
                  <div className={`h-6 w-6 mt-1 rounded-full ${isViolation ? "bg-violation-red" : "bg-compliance-green"}`}></div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-foreground">{c.name}</h4>
                      <Badge 
                        variant={isViolation ? "destructive" : "secondary"}
                        className={isViolation ? "bg-violation-red text-white" : "bg-compliance-green text-white"}
                      >
                        {c.status}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">{c.details}</p>
                    {c.policy_reference && (
                      <p className="text-xs text-muted-foreground mt-1">Ref: {c.policy_reference}</p>
                    )}
                  </div>
                </div>
              </Card>
            );
          })}
        </div>

        {/* Action Buttons */}
        {!noBillboard && violationCount > 0 && (
          <Card className="p-6 shadow-elevated animate-fade-in">
            <div className="text-center space-y-4">
              <h3 className="text-lg font-semibold text-foreground">Violations Detected</h3>
              <p className="text-sm text-muted-foreground">
                {violationCount} violation{violationCount > 1 ? 's' : ''} found. Submit a report to authorities?
              </p>
              <Button 
                onClick={() => onNavigate("report")}
                className="w-full bg-violation-red hover:bg-violation-red/90 text-lg py-6 rounded-xl transition-spring"
              >
                <FileText className="h-5 w-5 mr-2" />
                File Report
              </Button>
              <Button 
                variant="outline" 
                onClick={() => onNavigate("home")}
                className="w-full"
              >
                Scan Another Billboard
              </Button>
            </div>
          </Card>
        )}

        {!noBillboard && violationCount === 0 && (
          <Card className="p-6 shadow-elevated animate-fade-in bg-compliance-green/5 border-compliance-green">
            <div className="text-center space-y-4">
              <CheckCircle className="h-12 w-12 text-compliance-green mx-auto" />
              <h3 className="text-lg font-semibold text-foreground">All Clear!</h3>
              <p className="text-sm text-muted-foreground">
                This billboard appears to be fully compliant with regulations.
              </p>
              <Button 
                onClick={() => onNavigate("home")}
                className="w-full bg-compliance-green hover:bg-compliance-green/90"
              >
                Scan Another Billboard
              </Button>
            </div>
          </Card>
        )}

        {noBillboard && (
          <Card className="p-6 shadow-elevated animate-fade-in">
            <div className="text-center space-y-4">
              <h3 className="text-lg font-semibold text-foreground">No Billboard Detected</h3>
              <p className="text-sm text-muted-foreground">Try moving closer, keeping the full billboard in frame, and ensuring good lighting.</p>
              <Button onClick={() => onNavigate("camera")} className="w-full">
                Go Back to Camera
              </Button>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};