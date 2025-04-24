import os
import socket
import threading
import redis
import pickle
import signal
import sys
import base64
import cv2
import numpy as np
import torch
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import subprocess
import json
from datetime import datetime
from ultralytics import YOLO

from model_manager import MLModelManager
from server_handler import start_servers, signal_handler

# Directory for storing models
MODEL_FOLDER = './models'
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Connect to Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Global variable to control the server's running state
# server_running = True


### --- MLModelManager Class ---


### --- TCPServer Class ---



### --- Signal Handler for Graceful Shutdown ---

# def signal_handler(sig, frame):
#     """Handle SIGINT (Ctrl+C) for graceful shutdown."""
#     print("\nTerminated servers process...")
#     global server_running
#     server_running = False
#     tcp_server.stop_server()
#     sys.exit(0)


### --- Main Execution ---

if __name__ == '__main__':
    # Set up the signal handler for graceful shutdown
    # signal.signal(signal.SIGINT, signal_handler)

    # Initialize the MLModelManager
    model_manager = MLModelManager()

    tcp_server = start_servers()

    print("✅ Server is running... Press Ctrl+C to stop.")

    # Handle graceful shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: tcp_server.stop_server() or sys.exit(0))
    # Start the RedisHandler in a separate daemon thread
    try:
        while True:
            pass
    except KeyboardInterrupt:
        tcp_server.stop_server()
        sys.exit(0)
    
    
    # print("Loading model...")
    # # Load the model
    # model_name = "yolo_best.pt"  # Replace with your saved model name
    # try:
    #     model_manager.load_model(model_name)
    # except Exception as e:
    #     print(f"Failed to load model: {e}")
    #     sys.exit(1)

