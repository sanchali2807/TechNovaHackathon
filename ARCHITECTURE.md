# Billboard Watch India - Architecture Documentation

## Overview

Billboard Watch India is a comprehensive AI-powered billboard violation detection system that enables citizens to report billboard violations through a mobile-first web application. The system uses advanced machine learning models to detect billboards and automatically checks compliance against municipal policies.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │     Backend      │    │   ML Models     │
│   (React/TS)    │◄──►│   (FastAPI)      │◄──►│   (YOLOv8/ONNX) │
│                 │    │                  │    │                 │
│ • Camera UI     │    │ • REST API       │    │ • Billboard     │
│ • Gallery       │    │ • File Storage   │    │   Detection     │
│ • Dashboard     │    │ • Compliance     │    │ • Presence      │
│ • Violations    │    │   Engine         │    │   Classification│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Technology Stack

### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **UI Components**: Radix UI + Tailwind CSS
- **State Management**: React Query for server state
- **Routing**: React Router DOM
- **Icons**: Lucide React

### Backend
- **Framework**: FastAPI (Python)
- **ML Framework**: YOLOv8 (Ultralytics) + ONNX Runtime
- **Image Processing**: OpenCV
- **Data Validation**: Pydantic
- **CORS**: FastAPI CORS Middleware

### Machine Learning
- **Primary Model**: YOLOv8 (Object Detection)
- **Fallback Model**: MobileNetV2 (ONNX - Binary Classification)
- **Confidence Threshold**: 70% for billboard acceptance
- **Area Coverage**: Minimum 10% of image area

## Project Structure

```
billboard-watch-india/
├── src/                          # Frontend React application
│   ├── components/
│   │   ├── camera/              # Camera capture and gallery upload
│   │   ├── dashboard/           # Public analytics dashboard
│   │   ├── home/               # Landing page
│   │   └── violations/         # Violation analysis results
│   ├── hooks/                  # Custom React hooks
│   ├── lib/                    # Utility functions
│   └── pages/                  # Main page components
├── backend/
│   ├── app/
│   │   ├── models/             # Pydantic schemas
│   │   ├── pipeline/           # ML inference pipelines
│   │   │   ├── yolov8_billboard.py    # YOLOv8 detection
│   │   │   ├── presence.py             # ONNX classification
│   │   │   └── mock.py                 # Development fallback
│   │   ├── services/           # Business logic
│   │   │   ├── compliance.py           # Policy compliance checks
│   │   │   ├── storage.py              # File handling
│   │   │   └── policy_loader.py        # Policy management
│   │   ├── config.py           # Application configuration
│   │   └── main.py             # FastAPI application
│   ├── models/                 # ML model files
│   │   └── presence_mobilenet.onnx
│   ├── training/               # Model training scripts
│   └── uploads/                # File storage directory
├── models/                     # Additional model storage
├── policies.json              # Compliance policy definitions
└── package.json              # Frontend dependencies
```

## Data Flow Architecture

### 1. Image Capture & Upload Flow

```
User Action → Camera/Gallery → Image Processing → API Upload → ML Analysis → Compliance Check → Results Display
```

**Detailed Steps:**
1. **Image Capture**: User captures photo via camera or selects from gallery
2. **Preprocessing**: Frontend resizes/converts image to JPEG (max 1280x720)
3. **Metadata Collection**: GPS coordinates, timestamp automatically added
4. **API Request**: POST to `/upload` endpoint with image + metadata
5. **File Storage**: Backend saves file with UUID filename
6. **ML Pipeline**: Image processed through detection models
7. **Compliance Analysis**: Results checked against policy rules
8. **Response**: Structured analysis returned to frontend

### 2. ML Detection Pipeline

```
Image Input → Model Selection → YOLOv8 Detection → Validation Rules → ONNX Fallback → Final Decision
```

**Model Selection Logic:**
- **Primary**: YOLOv8 object detection (configurable via `model_provider`)
- **Fallback**: ONNX MobileNetV2 binary classification
- **Mock**: Development mode with simulated results

**Detection Validation:**
- Confidence threshold: ≥70%
- Area coverage: ≥10% of image
- Structure validation: Rectangular aspect ratio (1:3 to 3:1)
- Class filtering: Excludes person/animal detections

### 3. Compliance Engine Flow

```
Detection Results → Size Check → Placement Check → Content Check → License Check → Final Report
```

**Compliance Checks:**
1. **Size Compliance**: Against municipal area limits
2. **Placement Compliance**: Distance from restricted zones
3. **Content Compliance**: Text analysis for prohibited content
4. **License Compliance**: QR code/license number presence

## API Architecture

### Core Endpoints

#### POST `/upload`
**Purpose**: Main endpoint for billboard analysis
**Input**: 
- `file`: Image file (multipart/form-data)
- `metadata`: JSON string with location/timestamp

**Response**: `AnalysisResponse` with:
- Detection results (bounding boxes, confidence)
- Compliance report (violations found)
- Status (`success`, `no_billboard`, `uncertain`)

#### GET `/health`
**Purpose**: Service health check
**Response**: `{"status": "ok"}`

#### GET `/validate_billboard`
**Purpose**: Manual compliance validation
**Parameters**: width, height, text, location, city, qr
**Response**: Compliance report without ML detection

### Request/Response Schemas

```typescript
// Upload Metadata
interface UploadMetadata {
  timestamp: string;
  location?: {
    latitude: number;
    longitude: number;
    city?: string;
  };
}

// Analysis Response
interface AnalysisResponse {
  file_id: string;
  filename: string;
  media_type: "image" | "video";
  detections: {
    billboard_count: number;
    estimated_area_sqft: number;
    bounding_boxes: BoundingBox[];
  };
  compliance: {
    overall_passed: boolean;
    checks: ComplianceCheck[];
  };
  status: "success" | "no_billboard" | "uncertain";
  message?: string;
}
```

## Machine Learning Architecture

### YOLOv8 Detection Pipeline

**Model Configuration:**
- **Input**: 224x224 RGB images
- **Architecture**: YOLOv8 nano (yolov8n.pt)
- **Classes**: Detects objects, filters for billboard-like structures
- **Confidence**: 70% minimum threshold
- **Validation**: Geometric and structural checks

**Detection Process:**
1. Image preprocessing (resize, normalize)
2. YOLO inference with low confidence threshold (0.25) for debugging
3. Filter detections by class (exclude persons/animals)
4. Apply confidence threshold (0.7)
5. Validate structure (aspect ratio, size)
6. Return highest-confidence detection

### ONNX Fallback Model

**Model Details:**
- **Architecture**: MobileNetV2 binary classifier
- **Input**: 224x224 RGB, normalized [0,1]
- **Output**: [no_billboard, billboard] probabilities
- **Threshold**: 60% billboard confidence for acceptance

**Use Cases:**
- Fallback when YOLOv8 unavailable
- Secondary validation for edge cases
- Development environment testing

## Frontend Architecture

### Component Hierarchy

```
App
├── Index (Main Router)
│   ├── HomePage (Landing/Navigation)
│   ├── CameraPage (Capture/Upload)
│   ├── ViolationsPage (Analysis Results)
│   ├── DashboardPage (Public Analytics)
│   └── ReportPage (Violation Reporting)
└── Navigation (Bottom Tab Bar)
```

### State Management

**Global State:**
- Analysis results stored in `useAnalysis` hook
- Shared between Camera → Violations pages
- Includes detection data + image preview

**Local State:**
- Camera permissions and stream management
- Upload progress and error handling
- Form validation and submission

### UI/UX Design Patterns

**Mobile-First Design:**
- Responsive grid layouts
- Touch-optimized controls
- Progressive Web App capabilities

**Visual Feedback:**
- Loading states with animations
- Success/error message styling
- Real-time camera preview

## Policy & Compliance System

### Policy Structure

```json
{
  "national_policies": {
    "model_outdoor_advertising_policy_2016": {
      "size_rules": { "max_area_sqft": 500 },
      "placement_rules": ["junction", "hospital"],
      "content_restrictions": ["tobacco", "alcohol"],
      "licensing_rules": ["qr_code_required"]
    }
  },
  "municipal_policies": {
    "delhi": { "max_area_sqft": 300 },
    "mumbai": { "max_area_sqft": 400 }
  }
}
```

### Compliance Engine

**Check Types:**
1. **Size**: Area/dimension limits by city
2. **Placement**: Distance from restricted zones
3. **Content**: Text analysis for prohibited content
4. **License**: QR code/license visibility

**Violation Detection:**
- Each check returns `compliant` or `violation` status
- Detailed violation descriptions provided
- Policy references included for legal context

## Security & Performance

### Security Measures
- CORS configuration for frontend domains
- File type validation (images only)
- Input sanitization via Pydantic
- UUID-based file naming (prevents enumeration)

### Performance Optimizations
- Image resizing on frontend (reduces upload size)
- Model caching (singleton pattern)
- Async file operations
- Efficient OpenCV preprocessing

### Error Handling
- Graceful ML model fallbacks
- Detailed error messages for debugging
- Client-side validation before upload
- Comprehensive logging for troubleshooting

## Deployment Architecture

### Development Setup
```bash
# Frontend
npm install && npm run dev

# Backend
cd backend
pip install -r requirements.txt
python -m app.main

# ML Models
python setup_yolov8.py  # Downloads YOLOv8 model
```

### Production Considerations
- **Frontend**: Static build deployment (Vite)
- **Backend**: ASGI server (Uvicorn/Gunicorn)
- **Models**: Pre-download and cache ML models
- **Storage**: Persistent volume for uploads
- **Monitoring**: Health checks and logging

## Configuration Management

### Environment Variables
```bash
BILLBOARD_GUARD_MODEL_PROVIDER=yolov8    # Model selection
BILLBOARD_GUARD_MUNICIPAL_AREA_LIMIT_SQFT=40.0
BILLBOARD_GUARD_STORAGE_ROOT=/app/uploads
BILLBOARD_GUARD_PRESENCE_ENABLED=true
```

### Model Configuration
- Configurable confidence thresholds
- Switchable model providers
- Policy file path configuration
- Storage directory customization

This architecture provides a scalable, maintainable system for AI-powered billboard violation detection with comprehensive policy compliance checking and citizen engagement features.
