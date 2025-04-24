import json
import socket
import sys
import threading
import signal
from model_manager import MLModelManager

# ✅ ตัวแปรควบคุมการทำงานของเซิร์ฟเวอร์
server_running = True
tcp_server = None  

class TCPServer:
    def __init__(self, host='0.0.0.0', port=5001):
        self.host = host
        self.port = port
        self.server_socket = None

    def handle_command(self, command, conn, model_manager):
        """📌 ดำเนินการตามคำสั่งที่ได้รับ"""
        response = {"error": "Unknown command"}

        try:
            # ✅ แปลงข้อความเป็น JSON
            try:
                data = json.loads(command)
            except json.JSONDecodeError:
                response = {"error": "Invalid JSON format"}
                conn.sendall((json.dumps(response) + "\n").encode('utf-8'))
                return
            
            cmd_type = data.get("command", "").lower()
            project_id = data.get("project_id", "").strip()  # รับค่า project_id
            
            # ✅ ตรวจสอบว่า project_id ถูกส่งมาหรือไม่
            if not project_id:
                response = {"error": "Missing project ID (project_id)."}
                conn.sendall((json.dumps(response) + "\n").encode('utf-8'))
                return

            if cmd_type == 'train':
                # print(f"🚀 Running Training Command for project: {project_id}")
                model_name = data.get("model_name")
                mode = data.get("mode")
                
                if not model_name:
                    response = {"error": "Missing model_name for training."}
                elif mode not in ["segment", "detect", "classify"]:
                    response = {"error": f"Invalid mode '{mode}'. Choose from: segment, detect, classify."}
                else:
                    def background_train():
                        try:
                            result = model_manager.train_model(project_id, model_name, mode)
                            response = {"status": "success", "data": result}
                        except Exception as e:
                            response = {"error": str(e)}
                        conn.sendall((json.dumps(response) + "\n").encode('utf-8'))

                    threading.Thread(target=background_train, daemon=True).start()
                    return

                print(f"📌 Training Response: {response}")

            elif cmd_type == 'evaluate':
                print(f"🔍 Running Evaluation for project: {project_id}")

                model_name = data.get("model_name")

                if not model_name:
                    response = {"error": "Missing model_name for evaluation."}
                else:
                    try:
                        result = model_manager.evaluate_model(project_id, model_name)  # ✅ ส่ง `project_id`
                        response = {"status": "success", "data": result}
                    except (ValueError, FileNotFoundError, RuntimeError) as e:
                        response = {"error": str(e)}

                print(f"📌 Evaluation Response: {response}")

            elif cmd_type == 'predict':
                print(f"🔍 Running Predict Command for project: {project_id}")
                image_name = (data.get("image_name") or "").strip()
                print(f"📌 Image Name: {image_name}")
                if not image_name:
                    response = {"error": "Missing image name. Use: {'command': 'predict', 'project_id': 'proj_001', 'image_name': 'image_001'}"}
                else:
                    try:
                        result = model_manager.predict_from_path(project_id, image_name)  # ✅ ส่ง `project_id`
                        response = {"status": "success", "data": result}
                    except FileNotFoundError as e:
                        response = {"error": str(e)}
                    except RuntimeError as e:
                        response = {"error": f"Prediction failed: {str(e)}"}
                    except Exception as e:
                        response = {"error": f"An unexpected error occurred: {str(e)}"}

                print(f"📌 Prediction Response: {response}")

            elif cmd_type == 'deploy':
                model_name = data.get("model_name")
                if not model_name:
                    response = {"error": "Missing model name for deploy."}
                else:
                    try:
                        model_manager.load_model(project_id, model_name)  # ✅ ส่ง `project_id`
                        response = {"status": "success", "message": f"Model '{model_name}' deployed successfully"}
                    except FileNotFoundError as e:
                        response = {"error": str(e)}
                    except Exception as e:
                        response = {"error": f"Failed to deploy model: {str(e)}"}

            else:
                response = {"error": "Invalid command"}

        except Exception as e:
            response = {"error": f"Error processing command: {str(e)}"}

        conn.sendall((json.dumps(response) + "\n").encode('utf-8'))
        
    def handle_client(self, conn):
        """📌 รับคำสั่งจาก Client และส่งไป `handle_command()`"""
        model_manager = MLModelManager()
        
        try:
            with conn:
                while server_running:
                    data = conn.recv(4096)
                    if not data:
                        break
                    command = data.decode('utf-8').strip()
                    self.handle_command(command, conn, model_manager)
        except Exception as e:
            print(f"Error handling client: {e}")

    def start_server(self):
        """📌 เริ่ม TCP Server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"✅ TCP server running on {self.host}:{self.port}")

        while server_running:
            try:
                conn, addr = self.server_socket.accept()
                print(f"📌 Connected by {addr}")
                threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Server error: {e}")
                break

    def stop_server(self):
        """📌 หยุด TCP Server"""
        global server_running
        server_running = False
        if self.server_socket:
            self.server_socket.close()
            print("✅ TCP server stopped.")

### --- Start Servers ---
def start_servers():
    """📌 เริ่ม TCP Server"""
    global tcp_server
    tcp_server = TCPServer()
    tcp_thread = threading.Thread(target=tcp_server.start_server, daemon=True)
    tcp_thread.start()

    print("✅ Server started successfully.")
    return tcp_server

### --- Handle Exit ---
def signal_handler(sig, frame):
    """📌 Handle SIGINT (Ctrl+C) for graceful shutdown."""
    print("\n🔴 Terminating server...")
    tcp_server.stop_server()
    sys.exit(0)

# ✅ Register Signal Handler
signal.signal(signal.SIGINT, signal_handler)
