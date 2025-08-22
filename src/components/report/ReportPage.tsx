import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { ArrowLeft, Send, MapPin, Clock, Camera, Shield, AlertTriangle } from "lucide-react";

interface ReportPageProps {
  onNavigate: (page: string) => void;
}

export const ReportPage = ({ onNavigate }: ReportPageProps) => {
  const [showPrivacyModal, setShowPrivacyModal] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formData, setFormData] = useState({
    reporterName: "",
    reporterEmail: "",
    additionalComments: "",
    agreeToTerms: false
  });

  const handleSubmit = () => {
    if (!formData.agreeToTerms) {
      setShowPrivacyModal(true);
      return;
    }
    
    setIsSubmitting(true);
    // Simulate API call
    setTimeout(() => {
      setIsSubmitting(false);
      onNavigate("dashboard");
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-background pb-20">
      {/* Header */}
      <div className="bg-card border-b border-border px-6 py-4 flex items-center">
        <Button 
          variant="ghost" 
          size="sm" 
          onClick={() => onNavigate("violations")}
          className="mr-3 p-2"
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-xl font-semibold text-foreground">File Report</h1>
      </div>

      <div className="px-6 py-6 space-y-6">
        {/* Auto-filled Data */}
        <Card className="p-6 shadow-card animate-slide-up">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center">
            <Shield className="h-5 w-5 mr-2 text-primary" />
            Report Details (Auto-filled)
          </h2>
          
          <div className="space-y-4">
            <div className="flex items-center space-x-3 p-3 bg-muted/50 rounded-lg">
              <MapPin className="h-5 w-5 text-urban-blue" />
              <div>
                <p className="text-sm text-muted-foreground">Location</p>
                <p className="font-medium text-foreground">Connaught Place, New Delhi</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3 p-3 bg-muted/50 rounded-lg">
              <Clock className="h-5 w-5 text-urban-blue" />
              <div>
                <p className="text-sm text-muted-foreground">Timestamp</p>
                <p className="font-medium text-foreground">{new Date().toLocaleString()}</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3 p-3 bg-muted/50 rounded-lg">
              <Camera className="h-5 w-5 text-urban-blue" />
              <div>
                <p className="text-sm text-muted-foreground">Evidence</p>
                <p className="font-medium text-foreground">Photo & AI Analysis Attached</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3 p-3 bg-violation-red/10 rounded-lg border border-violation-red/20">
              <AlertTriangle className="h-5 w-5 text-violation-red" />
              <div>
                <p className="text-sm text-muted-foreground">Violations Detected</p>
                <p className="font-medium text-violation-red">3 violations found</p>
              </div>
            </div>
          </div>
        </Card>

        {/* Reporter Information */}
        <Card className="p-6 shadow-card animate-slide-in">
          <h3 className="text-lg font-semibold text-foreground mb-4">Reporter Information (Optional)</h3>
          
          <div className="space-y-4">
            <div>
              <Label htmlFor="name" className="text-sm font-medium text-foreground">
                Full Name
              </Label>
              <Input
                id="name"
                placeholder="Enter your name (optional)"
                value={formData.reporterName}
                onChange={(e) => setFormData({...formData, reporterName: e.target.value})}
                className="mt-1"
              />
            </div>
            
            <div>
              <Label htmlFor="email" className="text-sm font-medium text-foreground">
                Email Address
              </Label>
              <Input
                id="email"
                type="email"
                placeholder="Enter your email (optional)"
                value={formData.reporterEmail}
                onChange={(e) => setFormData({...formData, reporterEmail: e.target.value})}
                className="mt-1"
              />
            </div>
            
            <div>
              <Label htmlFor="comments" className="text-sm font-medium text-foreground">
                Additional Comments
              </Label>
              <Textarea
                id="comments"
                placeholder="Add any additional details about the violation..."
                value={formData.additionalComments}
                onChange={(e) => setFormData({...formData, additionalComments: e.target.value})}
                className="mt-1 min-h-[100px]"
              />
            </div>
          </div>
        </Card>

        {/* Privacy & Terms */}
        <Card className="p-6 shadow-card animate-fade-in">
          <div className="space-y-4">
            <div className="flex items-start space-x-3">
              <Checkbox
                id="terms"
                checked={formData.agreeToTerms}
                onCheckedChange={(checked) => 
                  setFormData({...formData, agreeToTerms: checked as boolean})
                }
                className="mt-1"
              />
              <div className="text-sm">
                <Label htmlFor="terms" className="text-foreground cursor-pointer">
                  I agree to submit this report to relevant authorities and understand that:
                </Label>
                <ul className="mt-2 text-muted-foreground space-y-1">
                  <li>• My report will be reviewed by municipal authorities</li>
                  <li>• Personal information is optional and protected</li>
                  <li>• False reports may have legal consequences</li>
                  <li>• Data is used solely for regulatory compliance</li>
                </ul>
              </div>
            </div>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowPrivacyModal(true)}
              className="text-xs"
            >
              View Full Privacy Policy
            </Button>
          </div>
        </Card>

        {/* Submit Button */}
        <Button
          onClick={handleSubmit}
          disabled={isSubmitting}
          className="w-full bg-primary hover:bg-primary/90 text-lg py-6 rounded-xl transition-spring"
        >
          {isSubmitting ? (
            <>
              <div className="animate-pulse-custom mr-2">⏳</div>
              Submitting Report...
            </>
          ) : (
            <>
              <Send className="h-5 w-5 mr-2" />
              Submit Report
            </>
          )}
        </Button>
      </div>

      {/* Privacy Modal */}
      {showPrivacyModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-6 z-50 animate-fade-in">
          <Card className="w-full max-w-md p-6 shadow-elevated max-h-[80vh] overflow-y-auto">
            <h3 className="text-lg font-semibold text-foreground mb-4">Privacy Policy</h3>
            <div className="text-sm text-muted-foreground space-y-3 mb-6">
              <p>Your privacy is important to us. This report will be used for:</p>
              <ul className="space-y-1">
                <li>• Municipal compliance enforcement</li>
                <li>• Urban planning improvements</li>
                <li>• Public safety assessments</li>
              </ul>
              <p>Personal information is optional and will be kept confidential.</p>
            </div>
            <div className="flex space-x-3">
              <Button
                variant="outline"
                onClick={() => setShowPrivacyModal(false)}
                className="flex-1"
              >
                Close
              </Button>
              <Button
                onClick={() => {
                  setFormData({...formData, agreeToTerms: true});
                  setShowPrivacyModal(false);
                }}
                className="flex-1"
              >
                I Agree
              </Button>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};