import { useState } from "react";
import { cn } from "@/lib/utils";
import { Home, Camera, AlertTriangle, FileText, BarChart3 } from "lucide-react";

interface NavigationProps {
  currentPage: string;
  onPageChange: (page: string) => void;
}

const navigationItems = [
  { id: "home", label: "Home", icon: Home },
  { id: "camera", label: "Scan", icon: Camera },
  { id: "violations", label: "Violations", icon: AlertTriangle },
  { id: "report", label: "Report", icon: FileText },
  { id: "dashboard", label: "Dashboard", icon: BarChart3 },
];

export const Navigation = ({ currentPage, onPageChange }: NavigationProps) => {
  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-card border-t border-border shadow-elevated z-50">
      <div className="flex items-center justify-around py-2 px-4 max-w-md mx-auto">
        {navigationItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;
          
          return (
            <button
              key={item.id}
              onClick={() => onPageChange(item.id)}
              className={cn(
                "flex flex-col items-center gap-1 p-2 rounded-lg transition-all duration-200",
                "min-w-0 flex-1",
                isActive
                  ? "bg-primary text-primary-foreground scale-105"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary"
              )}
            >
              <Icon className="h-5 w-5" />
              <span className="text-xs font-medium truncate">{item.label}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
};