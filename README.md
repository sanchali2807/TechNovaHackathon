# Billboard Watch India

AI-powered billboard detection and compliance monitoring system for outdoor advertising in India.

## Features

- **Billboard Detection**: YOLOv8-based detection to identify billboards vs non-billboard content
- **Compliance Monitoring**: Automated checks for advertising regulations
- **Real-time Analysis**: Upload images for instant billboard validation
- **Modern UI**: React-based interface with camera integration

## Setup & Installation

### Prerequisites
- Node.js & npm (for frontend)
- Python 3.8+ (for backend)
- Git

### Frontend Setup
```sh
# Clone the repository
git clone https://github.com/sanchali2807/TechNovaHackathon.git
cd TechNovaHackathon

# Install frontend dependencies
npm install

# Start the development server
npm run dev
```

### Backend Setup
```sh
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Running the Project

1. **Start Backend Server**:
   ```sh
   cd backend
   source .venv/bin/activate  # Activate virtual environment
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start Frontend Server** (in new terminal):
   ```sh
   npm run dev
   ```

3. **Access Application**:
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000

## Technologies Used

### Frontend
- **Vite** - Build tool
- **TypeScript** - Type safety
- **React** - UI framework
- **Tailwind CSS** - Styling
- **shadcn-ui** - Component library

### Backend
- **FastAPI** - Python web framework
- **YOLOv8** - Object detection model
- **OpenCV** - Image processing
- **Ultralytics** - YOLO implementation
- **Uvicorn** - ASGI server

## API Endpoints

- `POST /upload` - Upload billboard image for detection
- `GET /health` - Health check endpoint

## Model Configuration

The system uses YOLOv8 for billboard detection with the following criteria:
- Accepts: Non-person/animal objects with ≥40% confidence and ≥0.3% area coverage
- Rejects: People, animals, and low-confidence detections
- Supports: Real billboard structures, signs, and advertising displays
