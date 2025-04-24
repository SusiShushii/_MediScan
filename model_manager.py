
from datetime import datetime
import glob
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tkinter import Image
import logging

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from db_manager import DatabaseManager 
from utils.logger import log_info, log_error,log_warning  

# ✅ ใช้ relative path ไปยัง `MedSight_Project`
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "MedSight_Project"))
BASE_WORKSPACE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "MediScan", "workspace"))

class MLModelManager:
    def __init__(self):
        self.current_model = None
        self.db = DatabaseManager(BASE_PROJECT_DIR, BASE_WORKSPACE_DIR)
        self.model_name = None
        log_info("MLModelManager initialized.")

    def prepare_workspace(self, project_id, mode):
        """🔍 เรียก `prepare_workspace.py` พร้อมส่ง `BASE_PROJECT_DIR` และ `BASE_WORKSPACE_DIR`"""
        
        project_source_path = os.path.join(BASE_PROJECT_DIR, project_id)

        # Check wheather `MedSight_Project` exists or not
        if not os.path.exists(project_source_path):
            log_error(f"Project '{project_id}' not found at {project_source_path}")
            raise FileNotFoundError(f"❌ Project '{project_id}' not found at {project_source_path}")

        log_info(f"Running prepare_workspace.py for project_id: {project_id}, mode: {mode}")

        result = subprocess.run(
            ['python', 'prepare_workspace.py', project_id , mode],
            capture_output=True,
            text=True
        )

        print(result.stdout)
        log_info(result.stdout)
        if result.returncode != 0:
            log_error(f"prepare_workspace.py failed: {result.stderr}")
            raise RuntimeError(f"❌ Failed to prepare workspace: {result.stderr}")

        return True

    def save_model(self, project_id , model_name , mode , result_dir , model_path , validation_metrics):
        """ Save Model and Training Result to Database"""
        project_path = os.path.join(BASE_PROJECT_DIR,  project_id,"models",model_name)
        os.makedirs(project_path, exist_ok=True)  # Create folder if it doesn't exist
        
        # Check the project path
        if not os.path.exists(project_path):
            log_warning(f"Project '{project_id}' not found at {project_path}")
            raise FileNotFoundError(f"❌ Project '{project_id}' not found at {project_path}")
        
        # Change the name Training Result to `train_{mode}`
        train_output_path = os.path.join(project_path, "training_result")
        log_info(f"Copied training results to: {train_output_path}")
        shutil.copytree(result_dir, train_output_path, dirs_exist_ok=True) # ✅ กำหนด path ปลายทางของ Training Output

        # ✅ กำหนด path ของ model ที่จะ save → `model.pt`
        final_model_path = os.path.join(project_path, "model.pt")

        if os.path.exists(model_path):
            shutil.copy(model_path, final_model_path)
            log_info(f"Copied model to: {final_model_path}")
        else:
            log_error(f"Model file not found: {model_path} (Skipping copy)")

        # ✅ บันทึก path ลงใน Database
        model_id = self.db.insert_model(project_id, model_name , mode, project_path , validation_metrics)
        log_info(f"Model saved in DB with ID: {model_id}")
        return {
            "status": "success",
            "message": "Training results and model copied successfully",
            "model_id": model_id,
            "train_output_path": train_output_path,
            "model_path": final_model_path,
            "model_name": model_name,
            "validation_metrics": validation_metrics
        }

        
    def load_model(self, project_id, model_name):
        """Load a specific model from the ./models folder."""
        log_info(f"Deploying model: {model_name} for project: {project_id}")

        model_path = os.path.join(BASE_PROJECT_DIR, project_id, "models", model_name, "model.pt").replace("\\", "/")
        self.model_name = model_name
        
        # ตรวจสอบว่าไฟล์โมเดลมีอยู่จริง
        if not os.path.exists(model_path):
            log_error(f"Model {model_name} not found in {model_path}.")
            raise FileNotFoundError(f"Model {model_name} not found in {model_path}.")

        try:
            print(f"Loading model: {model_path}")
            # เลือกอุปกรณ์ (GPU ถ้ามี)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.current_model = YOLO(model_path).to(device)  # Use Ultralytics' YOLO to load the model
            load_result = (f"Model loaded successfully from {model_path}.")
            load_result = {
            "status": "success",
            "message": "Model loaded successfully",
            "model_path": model_path
            }
            print(load_result)
            log_info(f"Deployed model: {model_name} successfully in project: {project_id}.")
            return load_result
        except Exception as e:
            log_error(f"Error loading model {model_path}: {str(e)}")
            raise RuntimeError(f"Error loading model {model_path}: {str(e)}")

    def train_model(self, project_id ,model_name , mode):
        """Train the YOLO model using yolo.py with specific mode (detect, segment, classify)."""

        if mode not in ["detect", "segment", "classify"]:
            log_error("Invalid mode provided.")
            raise ValueError(f"Invalid mode '{mode}'. Use: detect, segment, classify")
        
        if self.db.model_exists(project_id, model_name):
            print(f"❌ Model name '{model_name}' already exists in project '{project_id}'")
            log_error("Model already exists in DB")
            raise ValueError(f"❌ Model name '{model_name}' already exists in project '{project_id}'")
        
        log_info(f"Training model: {model_name} in project: {project_id}, mode: {mode}")

        # Call `prepare_workspace.py` before training
        self.prepare_workspace(project_id , mode)
        
        project_path = os.path.join(BASE_WORKSPACE_DIR, project_id)

        # ✅ เช็คว่า project_path มีอยู่จริงไหม
        if not os.path.exists(project_path):
            log_error(f"Data path not found: {data_path} at {project_path}")
            raise FileNotFoundError(f"❌ Project '{project_id}' not found at {project_path}")

        if mode == "classify":
            data_path = os.path.join(project_path, "classification")  # ชี้ไปที่โฟลเดอร์
        else:
            data_path  = os.path.join(project_path, "data.yaml")  # segmentation/detection ใช้ yaml
        
        # ✅ เช็คเฉพาะ segmentation/detection เท่านั้น
        if mode != "classify" and not os.path.exists(data_path ):
            log_error(f"data.yaml not found in {data_path }")
            raise FileNotFoundError(f"❌ data.yaml not found in {data_path }")
        
        print("📂 Project Path:", project_path+"\n Data Path: ",data_path )
        print(f"🚀 Starting YOLO {mode} for PID: {project_id}")
        print(f" Data Path: ",data_path )
        try:
            # ✅ เรียก YOLO จริง
            result = subprocess.run(
                ['python', 'yolo.py', '--mode', mode, '--task', 'train','--data', data_path ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            print(f"📝 Raw Output from yolo.py:\n{stdout}")

            if result.returncode != 0:
                log_error(f"YOLO train failed: {result.stderr}")
                raise RuntimeError(f"❌ Training failed: {stderr}")

            # ✅ ค้นหา JSON Output ที่เป็นบรรทัดสุดท้าย
            json_lines = [line for line in stdout.splitlines() if line.strip().startswith('{') and line.strip().endswith('}')]

            if not json_lines:
                log_error("No valid JSON output found from YOLO")
                raise RuntimeError("❌ No valid JSON output found from yolo.py!")

            train_results = json.loads(json_lines[-1])  # ✅ **แปลง JSON String → Dict**
            
            # ✅ Debug: ตรวจสอบว่าค่าถูกต้อง
            print("Parsed JSON from YOLO:", train_results)

            if train_results.get("message") != "Training completed successfully":
                raise RuntimeError(f"❌ Training failed: {train_results.get('error', 'Unknown error')}")

            result_dir = train_results.get("result_dir")
            model_path = train_results.get("model_path")
            validation_metrics = train_results.get("validation_metrics", {})

            if not result_dir or not model_path:
                log_error("Missing training result paths")
                raise RuntimeError("❌ Training completed but result directory or model path not found.")
            
            # ✅ บันทึกโมเดลและอัปเดต DB
            save_result = self.save_model(project_id , model_name , mode , result_dir , model_path , validation_metrics)
            log_info(f"Evaluation results inserted for model {model_name} in project {project_id}")
            return save_result

        except Exception as e:
            log_error(f"An unexpected error occurred: {str(e)}")
            return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}


    def evaluate_model(self, project_id , model_name ):
        """Evaluate a specific YOLO model using yolo.py."""
        
        # ดึงข้อมูล model_id และ mode
        model_info = self.db.get_model_info(project_id, model_name)
        if not model_info:
            log_error(f"Model '{model_name}' not found in DB for project '{project_id}'")
            raise ValueError(f"❌ Model '{model_name}' not found in DB for project '{project_id}'")
        
        model_id = model_info["model_id"]
        mode = model_info["mode"]

        model_path = os.path.join(BASE_PROJECT_DIR, project_id, "models", model_name, "model.pt").replace("\\", "/")

        # Path dataset
        if mode in ["segment", "detect"]:
            data_path = os.path.join(BASE_WORKSPACE_DIR, project_id, "data.yaml").replace("\\", "/")
        elif mode == "classify":
            data_path = os.path.join(BASE_WORKSPACE_DIR, project_id, "classification").replace("\\", "/")
        else:
            raise ValueError("Invalid mode. Choose from: segment, detect, classify.")

        # ตรวจสอบว่าไฟล์โมเดลมีอยู่จริง
        if not os.path.exists(model_path):
            log_error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        log_info(f"Evaluating YOLO {mode} model (ID: {model_name}) for project {project_id}.")
        # เรียกใช้ `yolo.py` พร้อมส่ง mode และ model path
        try:
            result = subprocess.run(
                ['python', 'yolo.py', '--mode', mode, '--task', 'evaluate', '--trained_model_path', model_path,'--data', data_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            stdout = result.stdout.decode("utf-8").strip()
            stderr = result.stderr.decode("utf-8")

            # Debug: แสดงผลลัพธ์ที่ได้จาก yolo.py
            print(f"Raw Output from yolo.py:\n{stdout}")

            # เช็คว่า subprocess รันสำเร็จหรือไม่
            if result.returncode != 0:
                log_error(f"YOLO evaluate failed: {result.stderr.decode()}")
                raise RuntimeError(f"YOLO evaluation failed: {stderr}")

            # ค้นหา JSON Output ที่ถูกต้อง
            
            json_lines = [line for line in stdout.splitlines() if line.strip().startswith('{') and line.strip().endswith('}')]
            if not json_lines:
                log_error("No JSON output from YOLO evaluation")
                raise RuntimeError("Evaluation completed, but could not parse JSON output from yolo.py")

            eval_results = json.loads(json_lines[-1])  # ✅ ใช้ JSON บรรทัดสุดท้ายที่ print ออกมา

            try:
                insert_message = self.db.insert_evaluation(project_id, model_name, model_id, mode, eval_results)
                log_info(f"Evaluation results inserted for model={model_id}, model_name={model_name}, mode={mode} in project={project_id}")
            except RuntimeError as insert_error:
                log_error(f"Failed to save evaluation result to DB: {str(insert_error)}")
            
            return eval_results

        except Exception as e:
            log_error(f"An unexpected error occurred during evaluation: {str(e)}")
            raise RuntimeError(f"An unexpected error occurred during evaluation: {str(e)}")

    def predict_from_path(self, project_id , image_name, confidence_threshold=0.2):
        log_info(f"Prediction started: project={project_id}, image={image_name}, model={self.model_name}")
        if not self.current_model:
            log_error("No model loaded. Prediction aborted.")
            raise RuntimeError("⚠️ No model loaded. Please load a model first.")

        # 🔍 ใช้ glob หาไฟล์ภาพโดยไม่ระบุ extension
        search_pattern = os.path.join(BASE_PROJECT_DIR, "images", image_name + ".*")
        matched_files = glob.glob(search_pattern)

        if not matched_files:
            log_error(f"Image not found for: {image_name} in images/ folder")
            raise FileNotFoundError(f"❌ Image not found for: {image_name} in images/ folder")

        # ✅ ใช้ไฟล์แรกที่เจอ (อาจมี .jpg, .png, etc...)
        image_path = matched_files[0].replace("\\", "/")

        try:
            log_info(f"Predicting from Image path: {image_path}. Running Prediction...")

            # Using GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            results = self.current_model.predict(source=image_path, save=True, device=device)

            confidence_scores = {}
            predict_result = "No Detection Found"
            bounding_boxes = []

            # ✅ กรณี Classification
            if results[0].probs is not None:
                class_names = results[0].names
                probabilities = results[0].probs.data.cpu().numpy()  # Convert to NumPy array

                # 🔹 คัดกรองค่า confidence ที่สูงกว่า `confidence_threshold`
                confidence_scores = {
                    class_names[i]: float(probabilities[i]) 
                    for i in range(len(class_names)) 
                    if probabilities[i] >= confidence_threshold
                }
                if confidence_scores:
                    # เลือก predict_result ที่ confidence score มากที่สุด
                    predict_result = max(confidence_scores, key=confidence_scores.get)
                    log_info(f"Classification complete. Predicted: {predict_result}")
                else:
                    log_info(f"Prediction done — No confident classification above threshold {confidence_threshold}.")

                self.db.insert_prediction(
                    project_id=project_id,
                    image_name=image_name,
                    model_name=self.model_name,
                    predict_result=predict_result,
                    prediction_data={
                        "confidence_scores": confidence_scores,
                    }
                )
                return {
                    "status": "success",
                    "message": "Classification completed.",
                    "predict_result": predict_result,
                    "confidence_scores": confidence_scores
                }

            # ✅ กรณี Detection / Segmentation        
            for box in results[0].boxes:
                confidence = float(box.conf.tolist()[0])
                class_id = int(box.cls.tolist()[0])
                class_name = results[0].names[class_id]
                xyxy = box.xyxy.tolist()[0]

                if confidence >= confidence_threshold:
                    if class_name not in confidence_scores:
                        confidence_scores[class_name] = []
                    confidence_scores[class_name].append(confidence)
                    bounding_boxes.append({
                        "class": class_name,
                        "confidence": confidence,
                        "box": {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]}
                    })

            if confidence_scores:
                predict_result = max(confidence_scores, key=lambda k: sum(confidence_scores[k]) / len(confidence_scores[k]))
                log_info(f"Detection result: {predict_result}, Bounding boxes: {len(bounding_boxes)}")
            else:
                log_info("Detection complete — No objects above confidence threshold.")

            self.db.insert_prediction(
                project_id=project_id,
                image_name=image_name,
                model_name=self.model_name,
                predict_result=predict_result,
                prediction_data={
                    "confidence_scores": confidence_scores,
                    "bounding_boxes": bounding_boxes
                }
            )
            return {
                "status": "success",
                "message": "Detection completed",
                "predict_result": predict_result,
                "confidence_scores": confidence_scores,
                "bounding_boxes": bounding_boxes  
            }

        except Exception as e:
            log_error(f"Error during prediction: {str(e)}", exc_info=True)
            return {"error": f"An error occurred during prediction: {str(e)}"}
if __name__ == "__main__":
    model_manager = MLModelManager()