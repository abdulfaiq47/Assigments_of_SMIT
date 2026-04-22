from flask import Flask, request, jsonify, send_file
import os
import uuid
import threading
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# =========================
# HF SAFE STORAGE
# =========================
UPLOAD_FOLDER = "/tmp/uploads"
OUTPUT_FOLDER = "/tmp/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Task tracking
tasks = {}

# =========================
# BACKGROUND PROCESS
# =========================
def process_video(task_id, input_path):
    try:
        tasks[task_id]["status"] = "processing"

        base = f"{OUTPUT_FOLDER}/{task_id}"

        test_csv = f"{base}_test.csv"
        interp_csv = f"{base}_interpolated.csv"
        perfect_csv = f"{base}_perfect.csv"
        output_video = f"{base}.mp4"

        # =========================
        # STEP 1: MAIN DETECTION
        # =========================
        os.system(f"python main.py {input_path} {test_csv}")

        # =========================
        # STEP 2: INTERPOLATION + PERFECT CSV
        # =========================
        os.system(f"python add_missing_data.py {test_csv} {interp_csv} {perfect_csv}")

        # =========================
        # STEP 3: VISUALIZATION VIDEO
        # =========================
        os.system(f"python visualize.py {interp_csv} {output_video} {input_path}")

        # =========================
        # SAVE RESULTS
        # =========================
        tasks[task_id]["csv"] = perfect_csv
        tasks[task_id]["video"] = output_video
        tasks[task_id]["status"] = "done"

    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)

# =========================
# UPLOAD
# =========================
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]

    task_id = str(uuid.uuid4())
    input_path = f"{UPLOAD_FOLDER}/{task_id}.mp4"

    file.save(input_path)

    tasks[task_id] = {
        "status": "queued",
        "input": input_path
    }

    # Run in background thread (IMPORTANT for HF)
    thread = threading.Thread(
        target=process_video,
        args=(task_id, input_path)
    )
    thread.start()

    return jsonify({
        "task_id": task_id,
        "status": "queued"
    })

# =========================
# STATUS CHECK
# =========================
@app.route("/status/<task_id>")
def status(task_id):
    if task_id not in tasks:
        return jsonify({"error": "Invalid task"}), 404

    return jsonify(tasks[task_id])

# =========================
# DOWNLOAD VIDEO
# =========================
@app.route("/download/<task_id>")
def download(task_id):
    if task_id not in tasks or "video" not in tasks[task_id]:
        return jsonify({"error": "Not ready"}), 404

    return send_file(tasks[task_id]["video"], as_attachment=True)

# =========================
# DOWNLOAD CSV
# =========================
@app.route("/csv/<task_id>")
def csv(task_id):
    if task_id not in tasks or "csv" not in tasks[task_id]:
        return jsonify({"error": "Not ready"}), 404

    return send_file(tasks[task_id]["csv"], as_attachment=True)

# =========================
# HEALTH CHECK
# =========================
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "HF-compatible ANPR API"
    })

# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)