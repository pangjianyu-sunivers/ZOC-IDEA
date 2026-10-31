import os
import datetime
from flask import Flask, request, jsonify
from werkzeug.serving import make_server

videos_root = "/data/videos"
app = Flask(__name__)


@app.get("/")
def index():
    return jsonify(
        {
            "service": "dryai-upload-api",
            "status": "ok",
            "upload_endpoint": "/upload_video",
            "health_endpoint": "/health",
        }
    ), 200


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route('/upload_video', methods=['POST'])
def upload_video():
    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_hour = now.hour
    current_minute = now.minute

    time_interval = (current_minute // 5) * 5
    time_folder_name = f"{current_hour:02d}-{time_interval:02d}"

    date_folder_path = os.path.join(videos_root, current_date)
    time_folder_path = os.path.join(date_folder_path, time_folder_name)
    os.makedirs(time_folder_path, exist_ok=True)

    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "No video file provided"}), 400

    video_filename = video_file.filename
    video_path = os.path.join(time_folder_path, video_filename)
    video_file.save(video_path)

    return jsonify({"message": f"Video saved to {video_path}"}), 200


if __name__ == "__main__":
    print("[INFO] Starting video upload API on port 5001...")
    server = make_server('0.0.0.0', 5001, app)
    server.serve_forever()
