## 🚀 How to Run the Project

The server can receive commands via both **TCP Socket** and **Swagger API**.

---

### 📦 1. Install Dependencies
``bash
pip install -r requirements.txt

### 🧪 2. (Optional) Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv mediscan_env

# Activate on Windows
mediscan_env\Scripts\activate

# Deactivate environment (if needed)
deactivate
``

### 🔌 3. Running TCP Server (Main Backend)
python main.py

### 📚 4. Running Swagger API Documentation
uvicorn swagger_api:app --reload --port 8000
#📎 Swagger UI:
👉 http://127.0.0.1:8000/docs
