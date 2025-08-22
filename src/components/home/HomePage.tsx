import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Camera, Shield, MapPin, Users } from "lucide-react";

interface HomePageProps {
  onNavigate: (page: string) => void;
}

export const HomePage = ({ onNavigate }: HomePageProps) => {
  return (
    <div className="min-h-screen bg-background pb-20">
      {/* Header */}
      <div className="gradient-primary text-white px-6 pt-12 pb-8 rounded-b-3xl">
        <div className="text-center animate-slide-up">
          <div className="flex items-center justify-center mb-4">
            <Shield className="h-12 w-12 text-white mr-3" />
            <h1 className="text-3xl font-bold">BillboardGuard</h1>
          </div>
          <p className="text-lg opacity-90">
            Protecting Indian cities from unauthorized billboards
          </p>
        </div>
      </div>

      <div className="px-6 -mt-6 space-y-6">
        {/* Main Action Card */}
        <Card className="gradient-card shadow-elevated p-6 animate-slide-up">
          <div className="text-center space-y-4">
            <div className="bg-primary/10 rounded-full p-4 w-20 h-20 mx-auto flex items-center justify-center">
              <Camera className="h-10 w-10 text-primary" />
            </div>
            <h2 className="text-2xl font-bold text-foreground">
              Report a Billboard
            </h2>
            <p className="text-muted-foreground">
              Scan and detect violations in real-time with AI-powered analysis
            </p>
            <Button 
              onClick={() => onNavigate("camera")}
              className="w-full bg-primary hover:bg-primary/90 text-lg py-6 rounded-xl transition-spring"
            >
              Start Scanning
            </Button>
          </div>
        </Card>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 gap-4">
          <Card className="p-4 shadow-card animate-slide-in">
            <div className="text-center space-y-2">
              <MapPin className="h-8 w-8 text-urban-blue mx-auto" />
              <div className="text-2xl font-bold text-foreground">1,247</div>
              <div className="text-sm text-muted-foreground">Reports Filed</div>
            </div>
          </Card>
          
          <Card className="p-4 shadow-card animate-slide-in">
            <div className="text-center space-y-2">
              <Users className="h-8 w-8 text-compliance-green mx-auto" />
              <div className="text-2xl font-bold text-foreground">892</div>
              <div className="text-sm text-muted-foreground">Active Citizens</div>
            </div>
          </Card>
        </div>

        {/* Quick Actions */}
        <Card className="p-6 shadow-card animate-fade-in">
          <h3 className="text-lg font-semibold mb-4 text-foreground">Quick Actions</h3>
          <div className="space-y-3">
            <Button 
              variant="outline" 
              className="w-full justify-start"
              onClick={() => onNavigate("dashboard")}
            >
              <MapPin className="h-4 w-4 mr-3" />
              View Violations Map
            </Button>
            <Button 
              variant="outline" 
              className="w-full justify-start"
              onClick={() => onNavigate("violations")}
            >
              <Shield className="h-4 w-4 mr-3" />
              Recent Detections
            </Button>
          </div>
        </Card>

        {/* Info Banner */}
        <div className="bg-muted/50 rounded-xl p-4 border border-border">
          <p className="text-sm text-muted-foreground text-center">
            <span className="font-medium">Privacy Protected:</span> All reports are anonymized and help make our cities better
          </p>
        </div>
      </div>
    </div>
  );
};