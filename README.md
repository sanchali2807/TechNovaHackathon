# How to run the frontend

# Clone the repository
git clone https://github.com/sanchali2807/TechNovaHackathon.git
cd TechNovaHackathon

# Install frontend dependencies
npm install

# Start the development server
npm run dev








# how to run the backend
 Navigate to backend directory
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


## Running the Project

1. **Start Backend Server**:

   cd backend
   source .venv/bin/activate  # Activate virtual environment
   uvicorn main:app --reload --host 0.0.0.0 --port 8000


2. **Start Frontend Server** (in new terminal):

   npm run dev


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


