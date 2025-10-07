import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Camera, MapPin, Clock, Zap, ArrowLeft } from "lucide-react";
import { useAnalysis } from "@/hooks/useAnalysis";

interface CameraPageProps {
  onNavigate: (page: string) => void;
}

export const CameraPage = ({ onNavigate }: CameraPageProps) => {
  const { setAnalysis } = useAnalysis();
  const [isScanning, setIsScanning] = useState(false);
  const [location, setLocation] = useState("Fetching location...");
  const [timestamp, setTimestamp] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [coords, setCoords] = useState<{ lat?: number; lng?: number }>({});
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const lat = Number(pos.coords.latitude.toFixed(6));
          const lng = Number(pos.coords.longitude.toFixed(6));
          setCoords({ lat, lng });
          setLocation(`${lat}, ${lng}`);
        },
        () => {
          setLocation("Location unavailable");
        }
      );
    } else {
      setLocation("Geolocation not supported");
    }

    const updateTime = () => {
      const now = new Date();
      setTimestamp(now.toLocaleString());
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      setError(null);
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError("Camera API not supported in this browser");
        return;
      }
      const tryStart = async (facing: "environment" | "user") => {
        const constraints: MediaStreamConstraints = {
          video: {
            facingMode: { ideal: facing },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;
        if (videoRef.current) {
          const video = videoRef.current;
          video.setAttribute("playsinline", "");
          // Safari/iOS specific attribute
          video.setAttribute("webkit-playsinline", "");
          video.setAttribute("muted", "");
          video.setAttribute("autoplay", "");
          video.srcObject = stream;
          if (video.readyState >= 1 && (video.videoWidth || 0) > 0) {
            setIsCameraReady(true);
          } else {
            await new Promise<void>((resolve) => {
              const onLoaded = () => {
                video.removeEventListener("loadedmetadata", onLoaded);
                resolve();
              };
              video.addEventListener("loadedmetadata", onLoaded, { once: true });
            });
            setIsCameraReady(true);
          }
          await video.play();
        }
      };
      try {
        await tryStart("environment");
      } catch (envErr) {
        // Fallback to user-facing camera
        await tryStart("user");
      }
    } catch (e: any) {
      const msg = e?.name === "NotAllowedError" ? "Camera permission denied" : e?.message || "Unable to access camera";
      setError(msg);
    }
  };

  const captureAndUpload = async (event?: React.MouseEvent) => {
    // Prevent accidental captures from scroll or other events
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    
    if (!videoRef.current || isScanning) return;
    setIsScanning(true);
    try {
      const canvas = document.createElement("canvas");
      const scale = Math.min(1280 / (video.videoWidth || 1280), 720 / (video.videoHeight || 720));
      const w = Math.max(1, Math.floor((video.videoWidth || 640) * (scale || 1)));
      const h = Math.max(1, Math.floor((video.videoHeight || 480) * (scale || 1)));
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas not supported");
      ctx.drawImage(video, 0, 0, w, h);
      const blob: Blob | null = await new Promise((resolve) => canvas.toBlob(resolve as BlobCallback, "image/jpeg", 0.92));
      if (!blob) throw new Error("Failed to capture image");
      const apiBase = import.meta.env.VITE_API_BASE || "/api";
      const form = new FormData();
      form.append("file", new File([blob], "capture.jpg", { type: "image/jpeg" }));
      const metadata = {
        timestamp: new Date().toISOString(),
        location: coords.lat && coords.lng ? { latitude: coords.lat, longitude: coords.lng, city: undefined } : undefined,
      };
      form.append("metadata", JSON.stringify(metadata));
      
      const res = await fetch(`${apiBase}/upload`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Upload failed: ${res.status} ${body || ""}`);
      }
      const data = await res.json();
      if (data?.status === "no_billboard" || !data?.detections || (data.detections.billboard_count || 0) === 0) {
        const message = data?.message || "No billboard detected. Reframe to include the full billboard and try again.";
        setError(message);
        return;
      }
      if (data?.status === "uncertain") {
        const message = data?.message || "Uncertain detection. Please capture a clearer photo.";
        setError(message);
        return;
      }
{{ ... }}
      // Show success message briefly before navigating
      if (data?.message && data.message.includes('detected with') && data.message.includes('% confidence')) {
        setError(data.message);
        setTimeout(() => {
          const preview = canvas.toDataURL("image/jpeg", 0.8);
          setAnalysis(data, preview);
          onNavigate("violations");
        }, 1500);
      } else {
        const preview = canvas.toDataURL("image/jpeg", 0.8);
        setAnalysis(data, preview);
        onNavigate("violations");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Scan failed");
    } finally {
      setIsScanning(false);
    }
  };

  const openGallery = (event?: React.MouseEvent) => {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    if (isScanning) return;
    setError(null);
    fileInputRef.current?.click();
  };

  const onFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    try {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      setIsScanning(true);
      const apiBase = import.meta.env.VITE_API_BASE || "/api";
      const form = new FormData();

      // Convert non-JPEG images (e.g., HEIC/PNG) to JPEG for broader backend compatibility
      let uploadBlob: Blob = file;
      if (!/^image\/jpeg$/i.test(file.type)) {
        const objectUrl = URL.createObjectURL(file);
{{ ... }}
        try {
          const img = await new Promise<HTMLImageElement>((resolve, reject) => {
            const image = new Image();
            image.onload = () => resolve(image);
            image.onerror = () => reject(new Error("Failed to load selected image"));
            image.src = objectUrl;
          });
          const maxW = 1280;
          const maxH = 720;
          const ratio = Math.min(maxW / (img.width || maxW), maxH / (img.height || maxH));
          const w = Math.max(1, Math.floor((img.width || maxW) * ratio));
          const h = Math.max(1, Math.floor((img.height || maxH) * ratio));
          const canvas = document.createElement("canvas");
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext("2d");
          if (!ctx) throw new Error("Canvas not supported");
          ctx.drawImage(img, 0, 0, w, h);
          const jpegBlob: Blob | null = await new Promise((resolve) => canvas.toBlob(resolve as BlobCallback, "image/jpeg", 0.92));
          if (!jpegBlob) throw new Error("Failed to convert image");
          uploadBlob = jpegBlob;
        } finally {
          URL.revokeObjectURL(objectUrl);
        }
      }
      form.append("file", new File([uploadBlob], "gallery.jpg", { type: "image/jpeg" }));
      const metadata = {
        timestamp: new Date().toISOString(),
        location: coords.lat && coords.lng ? { latitude: coords.lat, longitude: coords.lng, city: undefined } : undefined,
      };
      form.append("metadata", JSON.stringify(metadata));
      const res = await fetch(`${apiBase}/upload`, { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Upload failed: ${res.status} ${body || ""}`);
      }
      const data = await res.json();
      if (data?.status === "no_billboard" || !data?.detections || (data.detections.billboard_count || 0) === 0) {
        const message = data?.message || "No billboard detected in the selected image. Please choose another where the billboard is clearly visible.";
        setError(message);
        return;
      }
      if (data?.status === "uncertain") {
        const message = data?.message || "Uncertain detection. Please choose a clearer image.";
        setError(message);
        return;
      }
      // Show success message briefly before navigating
      if (data?.message && data.message.includes('detected with') && data.message.includes('% confidence')) {
        setError(data.message);
        setTimeout(() => {
          // Build preview from the already selected file
          let preview: string | null = null;
          if (file) {
            const reader = new FileReader();
            reader.onload = () => {
              preview = String(reader.result);
              setAnalysis(data, preview);
              onNavigate("violations");
            };
            reader.readAsDataURL(file);
          } else {
            setAnalysis(data, preview);
            onNavigate("violations");
          }
        }, 1500);
      } else {
        // Build preview from the already selected file
        let preview: string | null = null;
        if (file) {
          preview = await new Promise<string>((resolve) => {
            const r = new FileReader();
            r.onload = () => resolve(String(r.result));
            r.readAsDataURL(file);
          });
        }
        setAnalysis(data, preview);
        onNavigate("violations");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsScanning(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div className="min-h-screen bg-background pb-20">
      {/* Header */}
      <div className="bg-card border-b border-border px-6 py-4 flex items-center">
        <Button 
          variant="ghost" 
          size="sm" 
          onClick={() => onNavigate("home")}
          className="mr-3 p-2"
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-xl font-semibold text-foreground">Scan Billboard</h1>
      </div>

      <div className="px-6 py-6 space-y-6">
        {/* Camera Viewfinder */}
        <Card className="relative overflow-hidden shadow-elevated animate-slide-up">
          <div className="aspect-[4/3] bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center relative">
            <div className="absolute inset-4 border-2 border-white/30 rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                className="w-full h-full object-cover rounded-md transition-opacity duration-300"
                autoPlay
                muted
                playsInline
                onLoadedMetadata={() => setIsCameraReady(true)}
                onCanPlay={() => setIsCameraReady(true)}
                style={{ opacity: isCameraReady ? 1 : 0.35 }}
              />
              {!isCameraReady && (
                <div className="absolute inset-0 flex items-center justify-center text-center text-white/70 bg-transparent">
                  <div>
                    <Camera className="h-16 w-16 mx-auto mb-4" />
                    <p className="text-lg">Enable camera to begin</p>
                    <p className="text-sm">Grant permission when prompted</p>
                  </div>
                </div>
              )}
            </div>
            
            {/* Scan overlay */}
            {isScanning && (
              <div className="absolute inset-0 bg-primary/20">
                <div className="absolute top-1/2 left-0 right-0 h-1 bg-primary animate-pulse"></div>
              </div>
            )}
          </div>
        </Card>

        {/* Location & Time Info */}
        <div className="grid grid-cols-1 gap-4">
          <Card className="p-4 shadow-card animate-slide-in">
            <div className="flex items-center space-x-3">
              <MapPin className="h-5 w-5 text-urban-blue" />
              <div>
                <p className="text-sm text-muted-foreground">Location</p>
                <p className="font-medium text-foreground">{location}</p>
              </div>
            </div>
          </Card>
          
          <Card className="p-4 shadow-card animate-slide-in">
            <div className="flex items-center space-x-3">
              <Clock className="h-5 w-5 text-urban-blue" />
              <div>
                <p className="text-sm text-muted-foreground">Timestamp</p>
                <p className="font-medium text-foreground">{timestamp}</p>
              </div>
            </div>
          </Card>
        </div>

        {/* Controls */}
        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {!isCameraReady ? (
              <Button
                onClick={startCamera}
                disabled={isScanning}
                className="w-full bg-primary hover:bg-primary/90 text-lg py-6 rounded-xl transition-spring"
              >
                <Camera className="h-5 w-5 mr-2" />
                Enable Camera
              </Button>
            ) : (
              <Button
                onClick={(e) => captureAndUpload(e)}
                disabled={isScanning}
                className="w-full bg-primary hover:bg-primary/90 text-lg py-6 rounded-xl transition-spring"
              >
                {isScanning ? (
                  <>
                    <Zap className="h-5 w-5 mr-2 animate-pulse" />
                    Scanning...
                  </>
                ) : (
                  <>
                    <Camera className="h-5 w-5 mr-2" />
                    Capture & Analyze
                  </>
                )}
              </Button>
            )}
            <Button
              variant="secondary"
              onClick={(e) => openGallery(e)}
              disabled={isScanning}
              className="w-full text-lg py-6 rounded-xl transition-spring"
            >
              Choose from Gallery
            </Button>
          </div>
          
          {/* Instructions */}
          <Card className="p-4 bg-muted/50 border border-border">
            <h3 className="font-medium text-foreground mb-2">Scanning Tips:</h3>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>• Ensure the entire billboard is visible</li>
              <li>• Take photo in good lighting conditions</li>
              <li>• Keep camera steady for best results</li>
              <li>• Include surrounding area for context</li>
            </ul>
          </Card>
          {error && (
            <Card className={`p-3 text-sm border ${
              error.includes('detected with') && error.includes('% confidence') 
                ? 'text-green-700 bg-green-50 border-green-200' 
                : 'text-red-600 bg-red-50 border-red-200'
            }`}>
              <div className="flex items-center space-x-2">
                {error.includes('detected with') && error.includes('% confidence') ? (
                  <span className="text-green-600">✅</span>
                ) : (
                  <span className="text-red-600">❌</span>
                )}
                <span>{error}</span>
              </div>
            </Card>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={onFileSelected}
            className="hidden"
          />
        </div>
      </div>
    </div>
  );
};