import os
import time
from datetime import datetime, timedelta
import requests
import json


def send_back_to_frontedge(pid, preds, preds_proba):
    """POST the prediction batch back to the front-end consumer."""
    url = "http://localhost:5002/return_result"
    # url = "http://0.0.0.0:5002/return_result"
    result_payload = {
        'pid': list(pid),
        'dryeye_label': list(preds),
        'dryeye_probs': list(preds_proba),
    }
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, data=json.dumps(result_payload), headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Error sending data: {e}"


# Root directory where uploaded videos are stored, bucketed by date and 5-minute window.
videos_root = "/data/videos"


def has_webm_videos(folder_path):
    """Return True if the 5-minute folder exists (used as a cheap presence check)."""
    if os.path.exists(folder_path):
        return True
    # for file in os.listdir(folder_path):
    #     if file.lower().endswith('.webm'):
    #         return True
    return False


def get_previous_5min(dt=None):
    """Return ``dt`` shifted 5 minutes backwards, defaulting to ``datetime.now()``."""
    if dt is None:
        dt = datetime.now()

    # # Floor the current minute to the nearest multiple of 5
    # rounded_minute = (dt.minute // 5) * 5
    # previous_interval = dt.replace(minute=rounded_minute, second=0, microsecond=0)

    # # Step one bucket back so we operate on a finished window
    # previous_interval -= timedelta(minutes=5)

    dt -= timedelta(minutes=5)
    print(dt)

    return dt


while True:
    print(datetime.now().minute)
    if datetime.now().minute % 5 == 0:
        print('yes')
        from datetime import datetime, timedelta
        from DE_system.blinking_detect import blinking_extract
        from DE_Vfeature_extract.blinking_analyze_new import video_feature_extract
        from DE_RedEye.red_eye_detect import red_eye_predict
        from DE_XGBoost.XGBoost_predict import predict_new_data
        # Move to the most recently finished 5-minute window.
        previous_5min = get_previous_5min()

        now = previous_5min
        current_date = now.strftime("%Y-%m-%d")
        current_hour = now.hour
        current_minute = now.minute

        # 5-minute bucket id
        time_interval = (current_minute // 5) * 5
        time_folder_name = f"{current_hour:02d}-{time_interval:02d}"

        # Resolve the bucket folder on disk
        date_folder_path = os.path.join(videos_root, current_date)
        time_folder_path = os.path.join(date_folder_path, time_folder_name)
        print(time_folder_path)


        if has_webm_videos(time_folder_path):

            print(f'working on videos on {time_folder_path}')

            # Stage 1 — extract per-frame eye-opening time series for every video in the bucket
            blinking_extract(time_folder_path)

            # Stage 2 — derive high-level blink features from the time series
            video_feature_extract(time_folder_path)

            # Stage 2.5 — run ResNet-50 on the max-eye-opening frame and write redness columns
            red_eye_predict(time_folder_path)

            # Stage 3 — feed blink + redness + OSDI features into the trained XGBoost classifier
            pid, preds, preds_proba = predict_new_data(time_folder_path)
            # print(pid)
            # print(preds)
            # print(preds_proba)

            # Stage 4 — POST the predictions back to the front-end
            # re_message = send_back_to_frontedge(pid,preds,preds_proba)
            print(send_back_to_frontedge(pid, preds, preds_proba))

        else:
            print('no videos upload')

    time.sleep(60)
