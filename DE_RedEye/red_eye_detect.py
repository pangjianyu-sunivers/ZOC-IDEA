"""Eye-redness detection stage (Stage 2.5).

For every video already processed by Stage 1, this module:
    1. Reads the per-frame eye-opening series produced by ``blinking_detect``
       from ``blinking_series.xlsx`` and locates the frame with the maximum
       eye opening (i.e. the most open-eye snapshot of the clip).
    2. Seeks that frame in the source video, runs dlib once to obtain the
       eye landmarks (points 36-47), and crops a padded bounding box that
       covers both eyes.
    3. Feeds the crop into a fine-tuned ResNet-50 binary classifier
       (red-eye vs. normal eye) and writes both the redness probability
       and the binary label back into ``blinking_results.xlsx`` so that
       Stage 3 (XGBoost) can pick it up as an extra input feature.
"""

import os

import cv2
import dlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


PREDICTOR_PATH = '/data/DE_system/shape_predictor_68_face_landmarks.dat'
MODEL_PATH = '/data/DE_RedEye/resnet50_redeye.pth'
SCALING = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_detector = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor(PREDICTOR_PATH)

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _load_model():
    """Load a ResNet-50 with a 2-class head from MODEL_PATH."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def _parse_blinking_series(value):
    """Convert the stringified list saved by Stage 1 back to a Python list."""
    if not isinstance(value, str):
        return []
    cleaned = value.strip('[]')
    if not cleaned:
        return []
    parts = cleaned.split(', ')
    if len(parts) < 2:
        return []
    try:
        return [float(p) for p in parts]
    except ValueError:
        return []


def _read_frame(video_path, frame_idx):
    """Seek to ``frame_idx`` in ``video_path`` and return a downscaled BGR frame."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return cv2.resize(frame, (0, 0), fx=SCALING, fy=SCALING)


def _crop_eye_region(frame, shape, expand=0.5):
    """Crop a padded bounding box around dlib eye landmarks 36-47."""
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(36, 48)])
    x, y, w, h = cv2.boundingRect(pts)
    pad_x = int(expand * w)
    pad_y = int(expand * h)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(frame.shape[1], x + w + pad_x)
    y1 = min(frame.shape[0], y + h + pad_y)
    return frame[y0:y1, x0:x1]


def red_eye_predict(folder_path):
    """Add ``eye_redness_prob`` / ``eye_redness_label`` columns into ``blinking_results.xlsx``.

    No-op (with a warning) when the ResNet-50 weights file is absent so that
    the rest of the pipeline can still run before the model is in place.
    """
    if not os.path.exists(MODEL_PATH):
        print(f'[red_eye_predict] Model file not found at {MODEL_PATH}, skipping.')
        return

    series_path = os.path.join(folder_path, 'blinking_series.xlsx')
    results_path = os.path.join(folder_path, 'blinking_results.xlsx')
    if not (os.path.exists(series_path) and os.path.exists(results_path)):
        print('[red_eye_predict] Required intermediate xlsx files missing, skipping.')
        return

    df_series = pd.read_excel(series_path)
    df_results = pd.read_excel(results_path)

    series_lookup = {
        row['video_name']: row['blinking']
        for _, row in df_series.iterrows()
        if isinstance(row.get('video_name'), str)
    }

    model = _load_model()

    for index, row in df_results.iterrows():
        video_name = row.get('video_name')
        if not isinstance(video_name, str):
            continue
        video_path = os.path.join(folder_path, video_name)
        if not os.path.exists(video_path):
            continue

        opening_series = _parse_blinking_series(series_lookup.get(video_name, ''))
        if not opening_series:
            continue
        max_idx = int(np.argmax(opening_series))

        frame = _read_frame(video_path, max_idx)
        if frame is None:
            continue
        dets = _detector(frame, 1)
        if not dets:
            continue
        shape = _predictor(frame, dets[0])
        eye_crop = _crop_eye_region(frame, shape)
        if eye_crop.size == 0:
            continue

        cv2.imwrite(
            os.path.join(folder_path, f'{video_name}_max_open_eye.png'),
            eye_crop,
        )

        rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
        x_in = _transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x_in)
            prob = torch.softmax(logits, dim=1)[0, 1].item()

        df_results.at[index, 'eye_redness_prob'] = prob
        df_results.at[index, 'eye_redness_label'] = int(prob >= 0.5)

    df_results.to_excel(results_path, index=False)
    print(f'[red_eye_predict] Eye-redness columns written to {results_path}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: python red_eye_detect.py <video_folder>')
        sys.exit(1)
    red_eye_predict(sys.argv[1])
