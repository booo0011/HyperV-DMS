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
)

if __name__ == '__main__':
    demo.launch()
