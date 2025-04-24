import os
import re
import torch
import json
import argparse
from ultralytics import YOLO
import numpy as np
from io import StringIO
import sys

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLO Model Operations")
    parser.add_argument('--mode', type=str, required=True, choices=['segment', 'detect', 'classify'],help="Model Type: segmentation, detection, or classification")
    parser.add_argument('--task', type=str, required=True, choices=['train', 'evaluate', 'test'],help="Task Type: train, evaluate, or test")
    parser.add_argument('--trained_model_path', type=str, required=False,help="Path to trained model (Only for evaluate/test)")
    parser.add_argument('--data', type=str, required=False, help="Path to dataset YAML file or classification folder")  # **เพิ่ม Argument**
    return parser.parse_args()

# Define Paths
BASE_PATH = "./"

MODEL_MAP = {
    "segment": "yolov8n-seg.pt",
    "detect": "yolov8n.pt",
    "classify": "yolov8n-cls.pt"
}

def train_model(mode,data_path):
    """Train YOLO model based on mode."""
    try:
        print(f"🚀 Starting YOLO {mode} training...")
        model = YOLO(MODEL_MAP[mode])
        print("Data Path in yolo_py: ",data_path)
        if mode == "classify":
            results = model.train(
                data=data_path,  
                epochs=100,
                imgsz=320,
                batch=32,
                amp=False,
            )
        else:
            results = model.train(
                data=data_path,
                epochs=100,
                imgsz=640,
                batch=32,
                amp=False,
            )

        print("\n✅ YOLO training completed!")
        print("===== YOLO Training Result Summary =====")

        # แสดงแค่ 20 บรรทัดสุดท้ายของ results
        # result_str_lines = str(results).split('\n')
        # for line in result_str_lines[-100:]:
        #     print(line)

        # ใช้ results.save_dir เพื่อหา path ที่ถูกต้อง
        save_dir = results.save_dir if hasattr(results, "save_dir") else None
        if save_dir:
            trained_model_path = os.path.join(str(save_dir), "weights", "best.pt")
            result_dir = str(save_dir)
        else:
            trained_model_path = None
            result_dir = None

        # ตรวจสอบว่ามีโมเดลหรือไม่
        if not trained_model_path or not os.path.exists(trained_model_path):
            print("Training completed, but no model was saved!")
            return {"error": "Model training completed, but model file not found."}
        
        result_dict = getattr(results, 'results_dict', {})
        class_names = getattr(results, 'names', {})
        print("Class Names from results.names:", class_names)
        if mode == "classify":
            # ย้ายตรงนี้มาใส่ใน result_json โดยตรง
            grouped_metrics = {
                "accuracy_top1": result_dict.get("metrics/accuracy_top1", 0.0),
                "accuracy_top5": result_dict.get("metrics/accuracy_top5", 0.0),
                "fitness": result_dict.get("fitness", 0.0)
            }

        else:
            grouped_metrics = {}
            other_metrics = {}
            index_map = {}
            class_tokens = set()

            for key in result_dict:
                match = re.match(r"metrics/([^()]+)\(([^()]+)\)", key)
                if match:
                    _, token = match.groups()
                    class_tokens.add(token)

            sorted_tokens = sorted(class_tokens)
            for idx, token in enumerate(sorted_tokens):
                index_map[token] = idx

            for key, value in result_dict.items():
                match = re.match(r"metrics/([^()]+)\(([^()]+)\)", key)
                if match:
                    metric, token = match.groups()
                    class_idx = index_map.get(token)
                    class_label = class_names.get(class_idx, f"class_{class_idx}")
                    if class_label not in grouped_metrics:
                        grouped_metrics[class_label] = {}
                    grouped_metrics[class_label][metric] = round(value, 4)
                else:
                    other_metrics[key] = round(value, 4) if isinstance(value, (float, int)) else value

            for label in grouped_metrics:
                token = sorted_tokens[list(class_names.values()).index(label)] if label in class_names.values() else None
                for key, value in other_metrics.items():
                    if token and f"({token})" in key and "mAP50-95" in key:
                        grouped_metrics[label]["mAP50-95"] = value

        result_json = {
            "message": "Training completed successfully",
            "mode": mode,
            "model_path": trained_model_path,
            "result_dir": result_dir,
            "validation_metrics": grouped_metrics
        }
        
        # Print JSON เป็นบรรทัดสุดท้าย
        print(json.dumps(result_json))
        return result_json

    except Exception as e:
        error_json = {"error": f"Training failed: {str(e)}"}
        print(json.dumps(error_json))
        return error_json


def evaluate_or_test_model(mode, trained_model_path, data_path, ):
    """Evaluate or test YOLO model."""
    try:
        eval_type = "test"
        print(f"\n🔹 Evaluating {mode} Model on {eval_type.upper()} Set...")
        model = YOLO(trained_model_path)

        if mode == "classify":
            # เลือก dataset ตาม `eval_type` (val หรือ test)
            split_type = "val" if eval_type == "val" else "test"
            metrics = model.val(data=data_path, split=split_type) # for classify

            # ค่าความถูกต้องทั่วไป (Overall)
            top1_accuracy = float(metrics.top1) if metrics.top1 is not None else 0.0
            top5_accuracy = float(metrics.top5) if metrics.top5 is not None else 0.0

            # ค่าความถูกต้องของแต่ละคลาส
            class_accuracies = {}
            confusion_matrix = metrics.confusion_matrix.matrix
            num_classes = confusion_matrix.shape[0]

            # ✅ คำนวณ Precision, Recall และ F1-score ของแต่ละคลาส
            class_metrics = {}

            for class_idx in range(num_classes):
                true_positive = confusion_matrix[class_idx, class_idx]  # ทำนายถูกต้อง
                false_positive = confusion_matrix[:, class_idx].sum() - true_positive  # ทำนายผิดว่าเป็น class นี้
                false_negative = confusion_matrix[class_idx, :].sum() - true_positive  # ทำนายผิดว่าไม่ใช่ class นี้
                total_samples = confusion_matrix[class_idx, :].sum()  # จำนวนตัวอย่างทั้งหมดของ class นี้
                
                # ✅ คำนวณ Accuracy ของแต่ละคลาส
                class_accuracy = true_positive / total_samples if total_samples > 0 else 0.0
                class_accuracies[f"class_{class_idx}"] = round(class_accuracy, 4)

                # ✅ คำนวณ Precision, Recall, F1-Score
                precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
                recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                class_metrics[f"class_{class_idx}"] = {
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1_score": round(f1_score, 4)
                }

            # ✅ คำนวณค่า Precision, Recall และ F1-score โดยเฉลี่ย (Macro Average)
            avg_precision = sum([m["precision"] for m in class_metrics.values()]) / num_classes
            avg_recall = sum([m["recall"] for m in class_metrics.values()]) / num_classes
            avg_f1_score = sum([m["f1_score"] for m in class_metrics.values()]) / num_classes

            # ใช้ JSON format 
            result_json = {
                "message": f"YOLO {eval_type} evaluation completed successfully",
                "metrics": {
                    "accuracy": round(top1_accuracy, 4),
                    "avg_precision": round(avg_precision, 4),
                    "avg_recall": round(avg_recall, 4),
                    "avg_f1_score": round(avg_f1_score, 4),
                    "class_accuracies": class_accuracies,
                    "class_metrics": class_metrics
                }
            }
            print(json.dumps(result_json))
            return result_json  # Return result to the caller
        else:
            
            results = model.val(data=data_path, split=eval_type, imgsz=640, conf=0.5) # # for detect/segment

            total_instances = results.box.nc  # จำนวนคลาสทั้งหมด
            # ค่า mAP (Mean Average Precision)
            map50 = results.box.map50  # mAP@50
            map95 = results.box.map  # mAP@50-95
            # Precision, Recall, และ F1 Score ทั้ง dataset
            overall_precision = np.array(results.box.p).mean()
            overall_recall =  np.array(results.box.r).mean()
            overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
            
            # เก็บค่าของแต่ละคลาส
            class_metrics = {}
            total_correct_predictions = 0  # ✅ ใช้คำนวณ Overall Accuracy
            total_instances = 0  # ✅ ใช้คำนวณ Overall Accuracy

            for idx, class_name in enumerate(results.names.values()):
                precision = results.box.p[idx].item() if idx < len(results.box.p) else 0
                recall = results.box.r[idx].item() if idx < len(results.box.r) else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (precision * recall) / (precision + recall - (precision * recall)) if (precision + recall) > 0 else 0  # 🔥 Accuracy Formula
                
                # คำนวณจำนวน Correct Predictions (TP) และ Instances
                class_instances = results.box.nc  # จำนวนตัวอย่างของคลาส
                correct_predictions = precision * recall * class_instances  # Approximate TP

                total_correct_predictions += correct_predictions
                total_instances += class_instances

                class_metrics[class_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "accuracy": accuracy  
                }
            
            # คำนวณ Overall Accuracy
            overall_accuracy = (total_correct_predictions / total_instances) if total_instances > 0 else 0

            # รวมค่าทั้งหมดไว้ใน JSON response
            result_json = {
                "message": f"YOLO {eval_type} evaluation completed successfully",
                "metrics": {
                    "mAP50": map50,
                    "mAP50-95": map95,
                    "overall_precision": overall_precision,
                    "overall_recall": overall_recall,
                    "overall_f1_score": overall_f1_score,
                    "overall_accuracy": overall_accuracy,
                    "class_details": class_metrics  # ข้อมูลของแต่ละคลาส
                }
            }
            print(json.dumps(result_json))
            return result_json  # Return result to the caller
    except Exception as e:
        error_json = {"error": f"Evaluation failed: {str(e)}"}
        print(json.dumps(error_json))
        return error_json  # Return error details to the caller

if __name__ == "__main__":
    args = parse_arguments()

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run based on the chosen task
    if args.task == "train":
        train_model(args.mode, args.data)
    elif args.task == "evaluate":
        if not args.trained_model_path:
            print(json.dumps({"error": "trained_model_path is required for evaluation"}))
        else:
            evaluate_or_test_model(args.mode, args.trained_model_path, args.data)