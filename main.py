from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# Heavy libraries will be lazy-loaded and cached to speed up --reload restarts
# Do NOT import face_recognition, numpy, or cv2 at module import time here
_face_recognition = None
_np = None
_cv2 = None

from PIL import Image
import io
import base64
import json
import os
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import uvicorn
import numpy.typing as npt

# Type hint for NumPy arrays (used in function signatures)
NDArray = Union[npt.NDArray[Any], Any]

import json, os

DATA_FILE = "data.json"

students_database = {}
attendance_records = []

def save_data():
    # Write to a temp file and atomically replace to avoid corruption
    tmp_file = DATA_FILE + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump({
            "students": students_database,
            "attendance": attendance_records
        }, f, indent=4)
    try:
        os.replace(tmp_file, DATA_FILE)
    except Exception:
        with open(DATA_FILE, "w") as f:
            json.dump({
                "students": students_database,
                "attendance": attendance_records
            }, f, indent=4)

def load_data():
    global students_database, attendance_records
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                students_database = data.get("students", {})
                attendance_records = data.get("attendance", [])
        except Exception:
            # If the data file is corrupt or can't be read, start with empty stores
            import traceback
            traceback.print_exc()
            students_database = {}
            attendance_records = []


def get_face_modules():
    """Lazy-import heavy image/face libraries and cache them.

    Returns tuple: (face_recognition, numpy, cv2)
    """
    global _face_recognition, _np, _cv2
    if _face_recognition is None or _np is None or _cv2 is None:
        try:
            # Import and cache the modules
            import face_recognition
            import numpy
            import cv2
            _face_recognition = face_recognition
            _np = numpy
            _cv2 = cv2
            print("Successfully loaded and cached face recognition modules")
        except Exception as e:
            print(f"Error loading face recognition modules: {e}")
            raise
    return _face_recognition, _np, _cv2


app = FastAPI(title="Student Attendance System", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (use database in production)
# We'll load existing data on startup so trained students persist across restarts


@app.on_event("startup")
def on_startup_load_data():
    """Load persisted students and attendance from disk on application startup."""
    load_data()

# Utility functions
def load_image_from_upload(file_content: bytes) -> NDArray:
    """Convert uploaded file to numpy array"""
    image = Image.open(io.BytesIO(file_content))
    # use numpy via cached getter to avoid top-level import
    _, np, _ = get_face_modules()
    return np.array(image)

def encode_image_to_base64(image: NDArray) -> str:
    """Convert numpy array to base64 string"""
    _, _, cv2 = get_face_modules()
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode()

def get_face_encodings(image: NDArray) -> tuple:
    """Extract face encodings and locations from image"""
    # Convert BGR to RGB if needed
    face_recognition, np, cv2 = get_face_modules()

    print(f"Image shape: {image.shape}, dtype: {image.dtype}")  # Debug: print image details
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Converted BGR to RGB")
    else:
        rgb_image = image
        print(f"Using image as-is (shape: {rgb_image.shape})")
    
    # Ensure image is uint8 (required by face_recognition)
    if rgb_image.dtype != np.uint8:
        rgb_image = rgb_image.astype(np.uint8)
        print("Converted image to uint8")

    # Find face locations and encodings
    face_locations = face_recognition.face_locations(rgb_image)
    print(f"Found {len(face_locations)} faces")  # Debug: print number of faces found
    
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        print(f"Generated {len(face_encodings)} face encodings")
    else:
        face_encodings = []
        print("No faces detected to encode")
    
    return face_encodings, face_locations

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Face Recognition API is running!",
        "version": "1.0.0",
        "endpoints": {
            "train": "/train - POST: Add student face",
            "recognize": "/recognize - POST: Recognize faces in image",
            "students": "/students - GET: List all students",
            "attendance": "/attendance - GET: Get attendance records",
            "clear_data": "/clear-data - DELETE: Clear all data"
        }
    }


@app.post("/warmup")
async def warmup():
    """Warm up heavy libraries so the first real request is fast.
    Call this once after a reload if you want to avoid a slow first recognition/train request.
    """
    try:
        get_face_modules()
        return JSONResponse({"status": "success", "message": "Warmup completed"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/train")
async def train_student_face(
    student_name: str = Form(...),
    student_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Train the system with a student's face (supports multiple images per student)
    - student_name: Name of the student
    - student_id: Unique ID for the student
    - file: Image file containing the student's face
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Load and process image
        image_content = await file.read()
        image = load_image_from_upload(image_content)
        
        # Get face encodings
        face_encodings, face_locations = get_face_encodings(image)
        
        if len(face_encodings) == 0:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        if len(face_encodings) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected. Please use an image with only one face for training")
        
        # Check if student already exists
        if student_id in students_database:
            # Add new encoding to existing student
            students_database[student_id]["encodings"].append(face_encodings[0].tolist())
            students_database[student_id]["face_locations"].append(face_locations[0])
            students_database[student_id]["last_trained"] = datetime.now().isoformat()
            total_images = len(students_database[student_id]["encodings"])

            # Persist updated student data
            try:
                save_data()
            except Exception:
                # Don't block response on save failures, but log
                import traceback
                traceback.print_exc()

            return JSONResponse({
                "status": "success",
                "message": f"Added new training image for {student_name}",
                "student_id": student_id,
                "total_images_for_student": total_images,
                "face_detected": True,
                "total_students": len(students_database)
            })
        else:
            # Create new student entry
            students_database[student_id] = {
                "name": student_name,
                "id": student_id,
                "encodings": [face_encodings[0].tolist()],  # List of encodings
                "face_locations": [face_locations[0]],  # List of face locations
                "first_trained": datetime.now().isoformat(),
                "last_trained": datetime.now().isoformat()
            }
            
            # Persist new student
            try:
                save_data()
            except Exception:
                import traceback
                traceback.print_exc()

            return JSONResponse({
                "status": "success",
                "message": f"Successfully created profile for {student_name}",
                "student_id": student_id,
                "total_images_for_student": 1,
                "face_detected": True,
                "total_students": len(students_database)
            })
        
    except Exception as e:
        import traceback
        print("Error in /train:", e)
        traceback.print_exc()  
        raise HTTPException(status_code=500, detail=f"Error processing image: {repr(e)}")

@app.post("/recognize")
async def recognize_faces(file: UploadFile = File(...), threshold: float = 0.55):
    """
    Recognize faces in an image and mark attendance
    - file: Image file containing faces to recognize
    - threshold: Recognition threshold (lower = more strict)
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Load and process image
        image_content = await file.read()
        image = load_image_from_upload(image_content)
        original_image = image.copy()
        
        # Get face encodings from uploaded image
        unknown_encodings, unknown_locations = get_face_encodings(image)
        
        if len(unknown_encodings) == 0:
            return JSONResponse({
                "status": "success",
                "message": "No faces detected in the image",
                "recognized_faces": [],
                "total_faces_detected": 0
            })
        
        # Prepare known faces data
        _, np, _ = get_face_modules()
        known_encodings = []
        known_names = []
        known_ids = []
        
        # For multiple encodings per student, we'll use all of them
        for student_id, student_data in students_database.items():
            for encoding in student_data["encodings"]:
                # Convert list to numpy array explicitly
                try:
                    encoding_array = np.array(encoding, dtype=np.float64)
                    known_encodings.append(encoding_array)
                    known_names.append(student_data["name"])
                    known_ids.append(student_id)
                except Exception as e:
                    print(f"Error converting encoding for student {student_id}: {e}")
                    continue  # Skip this encoding if conversion fails
        
        recognized_faces = []
        attendance_marked = []
        
        # Process each detected face
        for i, (unknown_encoding, face_location) in enumerate(zip(unknown_encodings, unknown_locations)):
            # Compare with known faces
            if known_encodings:
                face_recognition, np, _ = get_face_modules()
                # Calculate distances one by one to handle any type mismatches
                face_distances = []
                for known_encoding in known_encodings:
                    if isinstance(known_encoding, list):
                        known_encoding = np.array(known_encoding)
                    if isinstance(unknown_encoding, list):
                        unknown_encoding = np.array(unknown_encoding)
                    try:
                        distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
                        face_distances.append(distance)
                    except Exception as e:
                        print(f"Error calculating face distance: {e}")
                        face_distances.append(float('inf'))  # Use infinity for failed comparisons
                face_distances = np.array(face_distances)
                
                # Group results by student_id for voting
                student_votes = {}
                for idx, (distance, name, student_id) in enumerate(zip(face_distances, known_names, known_ids)):
                    if student_id not in student_votes:
                        student_votes[student_id] = {
                            'name': name,
                            'distances': [],
                            'confidences': []
                        }
                    student_votes[student_id]['distances'].append(distance)
                    student_votes[student_id]['confidences'].append(1 - distance)
                
                # Find best match using average distance
                best_student_id = None
                best_confidence = 0.0
                best_avg_distance = float('inf')
                
                for student_id, data in student_votes.items():
                    avg_distance = np.mean(data['distances'])
                    avg_confidence = np.mean(data['confidences'])
                    
                    if avg_distance < best_avg_distance:
                        best_avg_distance = avg_distance
                        best_confidence = avg_confidence
                        best_student_id = student_id
                
                if best_avg_distance < threshold:
                    # Face recognized
                    student_name = student_votes[best_student_id]['name']
                    confidence = best_confidence
                    
                    # Draw bounding box on image
                    top, right, bottom, left = face_location
                    _, _, cv2 = get_face_modules()  # Get cv2 for drawing
                    cv2.rectangle(original_image, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(original_image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    
                    # Show number of training images used
                    num_images = len(students_database[best_student_id]['encodings'])
                    cv2.putText(original_image, f"{student_name} ({confidence:.2f}) [{num_images}img]", 
                              (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                    
                    face_data = {
                        "student_id": best_student_id,
                        "student_name": student_name,
                        "confidence": float(confidence),
                        "training_images_used": num_images,
                        "average_distance": float(best_avg_distance),
                        "face_location": {
                            "top": int(top),
                            "right": int(right),
                            "bottom": int(bottom),
                            "left": int(left)
                        },
                        "status": "recognized"
                    }
                    
                    # Mark attendance (avoid duplicates)
                    if not any(record["student_id"] == best_student_id for record in attendance_marked):
                        attendance_record = {
                            "student_id": best_student_id,
                            "student_name": student_name,
                            "timestamp": datetime.now().isoformat(),
                            "confidence": float(confidence),
                            "training_images_used": num_images,
                            "status": "present"
                        }
                        attendance_records.append(attendance_record)
                        attendance_marked.append(attendance_record)

                        # Persist attendance immediately
                        try:
                            save_data()
                        except Exception:
                            import traceback
                            traceback.print_exc()
                    
                else:
                    # Face not recognized
                    top, right, bottom, left = face_location
                    _, _, cv2 = get_face_modules()  # Get cv2 for drawing
                    cv2.rectangle(original_image, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(original_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    cv2.putText(original_image, f"Unknown ({best_avg_distance:.2f})", 
                              (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                    
                    face_data = {
                        "student_id": None,
                        "student_name": "Unknown",
                        "confidence": float(1 - best_avg_distance),
                        "average_distance": float(best_avg_distance),
                        "face_location": {
                            "top": int(top),
                            "right": int(right),
                            "bottom": int(bottom),
                            "left": int(left)
                        },
                        "status": "unknown"
                    }
            else:
                # No trained faces available
                top, right, bottom, left = face_location
                _, _, cv2 = get_face_modules()  # Get cv2 for drawing
                cv2.rectangle(original_image, (left, top), (right, bottom), (255, 0, 0), 2)
                face_data = {
                    "student_id": None,
                    "student_name": "No trained data",
                    "confidence": 0.0,
                    "face_location": {
                        "top": int(top),
                        "right": int(right),
                        "bottom": int(bottom),
                        "left": int(left)
                    },
                    "status": "no_training_data"
                }
            
            recognized_faces.append(face_data)
        
        # Convert processed image to base64
        processed_image_base64 = encode_image_to_base64(original_image)
        
        return JSONResponse({
            "status": "success",
            "message": f"Processed {len(unknown_encodings)} faces",
            "total_faces_detected": len(unknown_encodings),
            "recognized_faces": recognized_faces,
            "attendance_marked": attendance_marked,
            "processed_image": processed_image_base64,
            "recognition_threshold": threshold
        })
        
    except Exception as e:
        import traceback
        print("Error in /recognize:", e)
        traceback.print_exc()  # Log the full traceback
        raise HTTPException(status_code=500, detail=f"Error recognizing faces: {repr(e)}")

@app.get("/students/{student_id}")
async def get_student_details(student_id: str):
    """Get detailed information about a specific student"""
    if student_id not in students_database:
        raise HTTPException(status_code=404, detail="Student not found")
    
    student_data = students_database[student_id]
    return JSONResponse({
        "status": "success",
        "student": {
            "student_id": student_id,
            "name": student_data["name"],
            "total_training_images": len(student_data["encodings"]),
            "first_trained": student_data["first_trained"],
            "last_trained": student_data["last_trained"],
            "training_complete": len(student_data["encodings"]) >= 3,
            "recommended_images": 4,
            "current_images": len(student_data["encodings"])
        }
    })

@app.get("/students")
async def get_students():
    """Get list of all trained students"""
    students_list = []
    for student_id, student_data in students_database.items():
        training_images = len(student_data["encodings"])
        students_list.append({
            "student_id": student_id,
            "name": student_data["name"],
            "total_training_images": training_images,
            "training_status": "complete" if training_images >= 3 else "needs_more_images",
            "first_trained": student_data["first_trained"],
            "last_trained": student_data["last_trained"]
        })
    
    return JSONResponse({
        "status": "success",
        "total_students": len(students_list),
        "students": students_list
    })

@app.get("/attendance")
async def get_attendance(date: Optional[str] = None):
    """
    Get attendance records
    - date: Optional date filter (YYYY-MM-DD format)
    """
    if date:
        filtered_records = [
            record for record in attendance_records 
            if record["timestamp"].startswith(date)
        ]
    else:
        filtered_records = attendance_records
    
    return JSONResponse({
        "status": "success",
        "total_records": len(filtered_records),
        "attendance_records": filtered_records,
        "filter_date": date
    })

@app.delete("/clear-data")
async def clear_all_data():
    """Clear all students and attendance data"""
    global students_database, attendance_records
    students_database.clear()
    attendance_records.clear()
    # Persist cleared state
    try:
        save_data()
    except Exception:
        import traceback
        traceback.print_exc()

    return JSONResponse({
        "status": "success",
        "message": "All data cleared successfully"
    })

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return JSONResponse({
        "status": "success",
        "stats": {
            "total_students": len(students_database),
            "total_attendance_records": len(attendance_records),
            "last_recognition": attendance_records[-1]["timestamp"] if attendance_records else None
        }
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)