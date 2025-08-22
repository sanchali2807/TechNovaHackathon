import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowLeft, MapPin, Trophy, AlertTriangle, TrendingUp, Users, Award } from "lucide-react";

interface DashboardPageProps {
  onNavigate: (page: string) => void;
}

const mockHeatmapData = [
  { area: "Connaught Place", violations: 23, status: "high" },
  { area: "Karol Bagh", violations: 15, status: "medium" },
  { area: "Lajpat Nagar", violations: 8, status: "low" },
  { area: "Chandni Chowk", violations: 31, status: "high" },
  { area: "Saket", violations: 12, status: "medium" },
];

const mockLeaderboard = [
  { rank: 1, name: "Priya S.", reports: 47, badge: "🥇" },
  { rank: 2, name: "Rajesh K.", reports: 39, badge: "🥈" },
  { rank: 3, name: "Amit P.", reports: 31, badge: "🥉" },
  { rank: 4, name: "Sneha M.", reports: 28, badge: "⭐" },
  { rank: 5, name: "Vikram R.", reports: 24, badge: "⭐" },
];

export const DashboardPage = ({ onNavigate }: DashboardPageProps) => {
  return (
    <div className="min-h-screen bg-background pb-20">
      {/* Header */}
      <div className="gradient-primary text-white px-6 pt-12 pb-6">
        <div className="flex items-center mb-4">
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={() => onNavigate("home")}
            className="mr-3 p-2 text-white hover:bg-white/20"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <h1 className="text-2xl font-bold">Public Dashboard</h1>
        </div>
        <p className="text-white/90">Real-time insights from citizen reports</p>
      </div>

      <div className="px-6 -mt-2 space-y-6">
        {/* Overall Stats */}
        <div className="grid grid-cols-3 gap-3 animate-slide-up">
          <Card className="p-4 shadow-card text-center">
            <AlertTriangle className="h-6 w-6 text-violation-red mx-auto mb-2" />
            <div className="text-lg font-bold text-foreground">1,247</div>
            <div className="text-xs text-muted-foreground">Total Reports</div>
          </Card>
          
          <Card className="p-4 shadow-card text-center">
            <TrendingUp className="h-6 w-6 text-urban-blue mx-auto mb-2" />
            <div className="text-lg font-bold text-foreground">89%</div>
            <div className="text-xs text-muted-foreground">Violation Rate</div>
          </Card>
          
          <Card className="p-4 shadow-card text-center">
            <Users className="h-6 w-6 text-compliance-green mx-auto mb-2" />
            <div className="text-lg font-bold text-foreground">892</div>
            <div className="text-xs text-muted-foreground">Active Citizens</div>
          </Card>
        </div>

        {/* Violations Heatmap */}
        <Card className="p-6 shadow-elevated animate-slide-in">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center">
            <MapPin className="h-5 w-5 mr-2 text-primary" />
            Violations Heatmap
          </h2>
          
          <div className="space-y-3">
            {mockHeatmapData.map((area, index) => (
              <div 
                key={area.area}
                className="flex items-center justify-between p-3 rounded-lg bg-muted/50 animate-slide-in"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="flex items-center space-x-3">
                  <div 
                    className={`w-3 h-3 rounded-full ${
                      area.status === "high" 
                        ? "bg-violation-red" 
                        : area.status === "medium" 
                        ? "bg-warning-orange" 
                        : "bg-compliance-green"
                    }`}
                  />
                  <span className="font-medium text-foreground">{area.area}</span>
                </div>
                
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-muted-foreground">
                    {area.violations} violations
                  </span>
                  <Badge 
                    variant={area.status === "high" ? "destructive" : "secondary"}
                    className={
                      area.status === "high" 
                        ? "bg-violation-red text-white" 
                        : area.status === "medium"
                        ? "bg-warning-orange text-white"
                        : "bg-compliance-green text-white"
                    }
                  >
                    {area.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-4 p-3 bg-muted/30 rounded-lg">
            <p className="text-xs text-muted-foreground text-center">
              🔴 High (20+) • 🟡 Medium (10-19) • 🟢 Low (&lt;10) violations per area
            </p>
          </div>
        </Card>

        {/* Citizen Leaderboard */}
        <Card className="p-6 shadow-elevated animate-fade-in">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center">
            <Trophy className="h-5 w-5 mr-2 text-warning-orange" />
            Top Contributors
          </h2>
          
          <div className="space-y-3">
            {mockLeaderboard.map((citizen, index) => (
              <div 
                key={citizen.rank}
                className={`flex items-center justify-between p-3 rounded-lg transition-smooth ${
                  citizen.rank <= 3 
                    ? "bg-gradient-to-r from-warning-orange/10 to-transparent border border-warning-orange/20" 
                    : "bg-muted/50"
                } animate-slide-in`}
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="flex items-center space-x-3">
                  <div className="text-lg">{citizen.badge}</div>
                  <div>
                    <p className="font-medium text-foreground">#{citizen.rank} {citizen.name}</p>
                    <p className="text-sm text-muted-foreground">{citizen.reports} reports filed</p>
                  </div>
                </div>
                
                {citizen.rank <= 3 && (
                  <Award className="h-5 w-5 text-warning-orange" />
                )}
              </div>
            ))}
          </div>
          
          <div className="mt-4 p-4 bg-gradient-to-r from-primary/10 to-transparent rounded-lg border border-primary/20">
            <div className="text-center space-y-2">
              <h3 className="font-semibold text-foreground">Join the Movement!</h3>
              <p className="text-sm text-muted-foreground">
                Report violations and climb the leaderboard. Every report helps make our cities better.
              </p>
              <Button 
                onClick={() => onNavigate("camera")}
                className="mt-3 bg-primary hover:bg-primary/90 transition-spring"
              >
                Start Reporting
              </Button>
            </div>
          </div>
        </Card>

        {/* Impact Stats */}
        <Card className="p-6 shadow-card animate-fade-in bg-gradient-to-br from-compliance-green/5 to-urban-blue/5 border border-compliance-green/20">
          <h3 className="text-lg font-semibold text-foreground mb-4 text-center">
            Community Impact
          </h3>
          
          <div className="grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-compliance-green">324</div>
              <div className="text-sm text-muted-foreground">Billboards Removed</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-urban-blue">₹2.4L</div>
              <div className="text-sm text-muted-foreground">Fines Collected</div>
            </div>
          </div>
          
          <p className="text-xs text-muted-foreground text-center mt-4">
            Thanks to citizen reporters, we've made real progress! 🎉
          </p>
        </Card>
      </div>
    </div>
  );
};