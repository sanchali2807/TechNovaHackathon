import { useState } from "react";
import { Navigation } from "@/components/ui/navigation";
import { HomePage } from "@/components/home/HomePage";
import { CameraPage } from "@/components/camera/CameraPage";
import { ViolationsPage } from "@/components/violations/ViolationsPage";
import { ReportPage } from "@/components/report/ReportPage";
import { DashboardPage } from "@/components/dashboard/DashboardPage";

const Index = () => {
  const [currentPage, setCurrentPage] = useState("home");

  const handlePageChange = (page: string) => {
    setCurrentPage(page);
  };

  const renderCurrentPage = () => {
    switch (currentPage) {
      case "home":
        return <HomePage onNavigate={handlePageChange} />;
      case "camera":
        return <CameraPage onNavigate={handlePageChange} />;
      case "violations":
        return <ViolationsPage onNavigate={handlePageChange} />;
      case "report":
        return <ReportPage onNavigate={handlePageChange} />;
      case "dashboard":
        return <DashboardPage onNavigate={handlePageChange} />;
      default:
        return <HomePage onNavigate={handlePageChange} />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {renderCurrentPage()}
      <Navigation currentPage={currentPage} onPageChange={handlePageChange} />
    </div>
  );
};

export default Index;
