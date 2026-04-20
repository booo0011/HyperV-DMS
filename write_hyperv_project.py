import os
import textwrap
import shutil

base = os.getcwd()
for d in ['assets', 'models', 'notebooks', 'core']:
    os.makedirs(os.path.join(base, d), exist_ok=True)

feature_code = textwrap.dedent('''
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


def _ensure_mediapipe_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def _extract_audio_mfcc(video_path, sr=16000, n_mfcc=40):
    audio_path = os.path.splitext(video_path)[0] + '.wav'
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error', '-i', video_path,
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
''')

with open(os.path.join(base, 'core', 'feature_extractor.py'), 'w', encoding='utf-8') as f:
    f.write(feature_code)

hgnn_code = textwrap.dedent('''
import torch
import torch.nn as nn

EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised']

class BatchedHGNNLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.W = nn.Linear(in_ch, out_ch)

    def forward(self, x, H):
        De = torch.sum(H, dim=1) + 1e-6
        Dv = torch.sum(H, dim=2) + 1e-6
        inv_De = torch.diag_embed(1.0 / De)
        inv_sqrt_Dv = torch.diag_embed(1.0 / torch.sqrt(Dv))
        out = torch.bmm(inv_sqrt_Dv, x)
        out = torch.bmm(H.transpose(1, 2), out)
        out = torch.bmm(inv_De, out)
        out = torch.bmm(H, out)
        out = torch.bmm(inv_sqrt_Dv, out)
        return self.W(out)

class ResumeVoyagerNet(nn.Module):
    def __init__(self, v_dim=956, a_dim=40, hidden_dim=128, num_hyperedges=4):
        super().__init__()
        self.v_proj = nn.Sequential(nn.Linear(v_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3))
        self.a_proj = nn.Sequential(nn.Linear(a_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3))
        self.H_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_hyperedges)
        )
        self.batched_hgnn = BatchedHGNNLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, 5)

    def forward(self, v_feat, a_feat):
        vh = self.v_proj(v_feat)
        ah = self.a_proj(a_feat)
        nodes = torch.stack([vh, ah], dim=1)
        logits = self.H_generator(nodes)
        H = torch.sigmoid(logits)
        fused_nodes = self.batched_hgnn(nodes, H)
        v_final = vh + fused_nodes[:, 0, :]
        a_final = ah + fused_nodes[:, 1, :]
        final_feat = torch.cat([v_final, a_final], dim=1)
        return self.classifier(final_feat)
''')

with open(os.path.join(base, 'core', 'hgnn_model.py'), 'w', encoding='utf-8') as f:
    f.write(hgnn_code)

app_code = textwrap.dedent('''
import os
import torch
import gradio as gr
from core.feature_extractor import process_multimodal_video
from core.hgnn_model import ResumeVoyagerNet, EMOTION_LABELS

MODEL_PATH = os.path.join('models', 'resume_voyagernet_epoch50.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResumeVoyagerNet().to(device)
if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
    except Exception as exc:
        print(f'Warning: failed to load model weights: {exc}')
else:
    print(f'Warning: model weights not found at {MODEL_PATH}. Predictions will use an uninitialized model.')


def predict_emotion(video_path):
    if video_path is None:
        return {label: 0.0 for label in EMOTION_LABELS}
    v_feat, a_feat = process_multimodal_video(video_path)
    with torch.no_grad():
        v_feat = v_feat.to(device)
        a_feat = a_feat.to(device)
        logits = model(v_feat, a_feat)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}

title = 'HyperV-DMS Driver Emotion Dashboard'
description = 'Upload a driver recording and preview the predicted emotion distribution.'

demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Video(label='Upload a driver video'),
    outputs=gr.Label(num_top_classes=3, label='Emotion Prediction'),
    title=title,
    description=description,
    allow_flagging='never',
)

if __name__ == '__main__':
    demo.launch()
''')

with open(os.path.join(base, 'app.py'), 'w', encoding='utf-8') as f:
    f.write(app_code)

requirements = textwrap.dedent('''
numpy<2.0
mediapipe
opencv-contrib-python
torch
torchvision
torchaudio
librosa
scikit-learn
matplotlib
seaborn
tqdm
gradio
requests
''')
with open(os.path.join(base, 'requirements.txt'), 'w', encoding='utf-8') as f:
    f.write(requirements)

readme = textwrap.dedent('''
# HyperV-DMS

HyperV-DMS is a refactored repository for the Hypergraph-based Multimodal Driver Monitoring System.

## Project structure
- `assets/` : demo GIFs and architecture diagrams.
- `models/` : trained weights, e.g. `resume_voyagernet_epoch50.pth`.
- `notebooks/` : training and EDA notebooks.
- `core/` : core algorithm modules.
- `app.py` : Gradio web demo entrypoint.
- `requirements.txt` : Python dependencies.

## Quick start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure `ffmpeg` is installed and available on your `PATH`.
3. Place your trained weights in `models/resume_voyagernet_epoch50.pth`.
4. Run the app:
   ```bash
   python app.py
   ```

## Notebook
The original notebook has been relocated to `notebooks/model_training_and_eda.ipynb`.
''')
with open(os.path.join(base, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(readme)

for placeholder in [os.path.join(base, 'assets', 'demo_angry_alert.gif'), os.path.join(base, 'models', 'resume_voyagernet_epoch50.pth')]:
    if not os.path.exists(placeholder):
        open(placeholder, 'a', encoding='utf-8').close()

src_nb = os.path.join(base, 'Voyah_HyperEmotion (3).ipynb')
dst_nb = os.path.join(base, 'notebooks', 'model_training_and_eda.ipynb')
if os.path.exists(src_nb) and not os.path.exists(dst_nb):
    shutil.move(src_nb, dst_nb)

print('finished')
