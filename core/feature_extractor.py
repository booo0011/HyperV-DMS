import os
import subprocess
import urllib.request

import cv2
import librosa
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_BASE_DIR, 'face_landmarker.task')
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

import imageio_ffmpeg


def _ensure_mediapipe_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def _extract_audio_mfcc(video_path, sr=16000, n_mfcc=40):
    audio_path = os.path.splitext(video_path)[0] + '.wav'
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_cmd = [
        ffmpeg_exe, '-y', '-loglevel', 'error', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', str(sr), '-ac', '1', audio_path
    ]
    subprocess.run(ffmpeg_cmd, capture_output=True)

    if not os.path.exists(audio_path):
        return torch.zeros((1, n_mfcc), dtype=torch.float32)

    y, _ = librosa.load(audio_path, sr=sr)
    if len(y) == 0:
        os.remove(audio_path)
        return torch.zeros((1, n_mfcc), dtype=torch.float32)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    os.remove(audio_path)
    return torch.from_numpy(np.mean(mfcc.T, axis=0)).unsqueeze(0).float()


def _extract_face_topology(video_path, max_faces=1):
    model_path = _ensure_mediapipe_model()
    cap = cv2.VideoCapture(video_path)
    v_list = []
    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=max_faces)
        with vision.FaceLandmarker.create_from_options(options) as detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = detector.detect(mp_img)
                if res.face_landmarks:
                    landmark = res.face_landmarks[0]
                    coords = np.array([[p.x, p.y] for p in landmark]).flatten()
                    v_list.append(coords)
    except Exception:
        pass
    finally:
        cap.release()

    if len(v_list) == 0:
        return torch.zeros((1, 956), dtype=torch.float32)
    return torch.from_numpy(np.mean(np.stack(v_list, axis=0), axis=0)).unsqueeze(0).float()


def process_multimodal_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video not found: {video_path}')
    v_feat = _extract_face_topology(video_path)
    a_feat = _extract_audio_mfcc(video_path)
    return v_feat, a_feat


def process_static_features(video_path):
    """Alias for process_multimodal_video; returns 1-D tensors (no batch dim)."""
    v_feat, a_feat = process_multimodal_video(video_path)
    return v_feat.squeeze(0), a_feat.squeeze(0)
