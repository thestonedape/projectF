# Face Recognition Student Attendance API

This project is a FastAPI-based server for face recognition and student attendance management. It uses OpenCV and face_recognition to detect and recognize faces from images, allowing you to train student profiles and recognize them via API endpoints.

## Features
- Train new students with face images
- Recognize students from uploaded images
- Persistent storage of trained students (data.json)
- FastAPI endpoints for easy integration
- Lazy loading for fast reloads

## Requirements
- Python 3.10+
- Windows (recommended, but works on Linux/Mac with minor changes)
- Recommended: Use a virtual environment

## Installation
1. **Clone or copy the project folder to your laptop.**
2. **Open a terminal in the project directory.**
3. **Create and activate a virtual environment (recommended):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
4. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## Usage
1. **Start the server:**
   ```powershell
   uvicorn main:app --reload
   ```
2. **API Endpoints:**
   - `POST /train` - Train a new student. Send a form with `name` and `file` (image).
   - `POST /recognize` - Recognize a student from an image. Send a form with `file` (image).
   - `GET /students` - List all trained students.

3. **Example (using curl):**
   ```powershell
   curl.exe -X POST -F "name=John Doe" -F "file=@train1.jpg" http://localhost:8000/train
   curl.exe -X POST -F "file=@test1.jpg" http://localhost:8000/recognize
   ```

## Data Persistence
- Trained students are saved in `data.json` in the project folder.
- Data is loaded automatically on server startup.
- To reset all students, delete `data.json` and restart the server.

## Notes
- Images should be clear, well-lit, and show only one face for best results.
- If you move the project to another laptop, copy the entire folder including `data.json` if you want to keep trained students.
- If you want a fresh start, delete `data.json` before running.

## Troubleshooting
- If you get errors about missing packages, run `pip install -r requirements.txt` again.
- For face recognition errors, ensure images are valid and contain faces.
- For performance issues, try restarting the server or using the warmup endpoint (if implemented).

## License
MIT

## Author
thestonedape
