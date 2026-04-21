import gradio as gr
import torch
import torch.nn.functional as F
import os

from core.hgnn_model import ResumeVoyagerNet
from core.feature_extractor import process_static_features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Search for a trained model in common locations
_model_candidates = [
    os.path.join(os.path.dirname(__file__), 'models', 'final_resume_voyagernet.pth'),
    os.path.join(os.path.dirname(__file__), 'models', 'resume_voyagernet_epoch50.pth'),
    '/content/models/final_resume_voyagernet.pth',
]
model_save_path = next((p for p in _model_candidates if os.path.exists(p)), _model_candidates[0])

inference_model = ResumeVoyagerNet(v_dim=956, a_dim=40, hidden_dim=128).to(device)
if os.path.exists(model_save_path):
    inference_model.load_state_dict(torch.load(model_save_path, map_location=device))
    print(f" 成功加载保存的模型权重: {model_save_path}")
else:
    print(f" 警告: 未找到 {model_save_path}，当前使用随机初始化模型。")

inference_model.eval()

def dms_inference(video_path, audio_path):
    if video_path is None and audio_path is None:
        return {"等待接入": 1.0}, '<div class="alert-box normal">系统就绪：请启动传感器或上传文件</div>'

    try:
        if video_path is not None and audio_path is None:
            audio_path = video_path

        if video_path is not None:
            v_feat, a_feat = process_static_features(video_path)
        else:
            v_feat, a_feat = torch.zeros(956), torch.zeros(40)

        v_feat = v_feat.unsqueeze(0).to(device)
        a_feat = a_feat.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = inference_model(v_feat, a_feat)
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        classes = ["Neutral (平稳)", "Happy (兴奋)", "Sad (低落)", "Angry (路怒)", "Surprised (惊吓)"]
        result_dict = {cls: prob for cls, prob in zip(classes, probs)}

        if result_dict["Angry (路怒)"] > 0.4:
            alert_html = """
            <div class="alert-box danger">
                <div class="alert-title">🚨 危险：路怒状态拦截</div>
                <div class="alert-subtitle">动力输出已限制 30% | AEB已激活</div>
            </div>
            """
        elif result_dict["Happy (兴奋)"] > 0.6:
            alert_html = """
            <div class="alert-box warning">
                <div class="alert-title">⚠️ 警告：驾驶员分心</div>
                <div class="alert-subtitle">请注视前方道路</div>
            </div>
            """
        else:
            alert_html = """
            <div class="alert-box safe">
                <div class="alert-title">✔️ 驾驶员状态平稳</div>
                <div class="alert-subtitle">HyperV-DMS 监控中</div>
            </div>
            """
        return result_dict, alert_html
    except Exception as e:
        return {"Error": 1.0}, f'<div class="alert-box danger">故障: {str(e)}</div>'

car_css = """
* { box-sizing: border-box !important; }
body, html {
    padding: 0 !important; margin: 0 !important;
    width: 100vw !important; height: 100vh !important;
    background-color: #000 !important; overflow: hidden !important;
}
.gradio-container, .gradio-container > .main, .gradio-container > .main > .wrap {
    max-width: 100vw !important; width: 100vw !important;
    padding: 0 !important; margin: 0 !important;
}
footer {display: none !important;}

.top-status-bar {
    position: fixed !important; top: 0 !important; left: 0 !important;
    width: 100vw !important; height: 40px !important;
    background: #fdfdfd; color: #333;
    display: flex; justify-content: space-between; align-items: center;
    padding: 0 40px; font-size: 13px; font-weight: 500;
    border-bottom: 1px solid #eee; z-index: 9999 !important;
}

.middle-wrapper {
    position: fixed !important; top: 40px !important; left: 0 !important;
    display: flex !important; width: 100vw !important; height: calc(100vh - 120px) !important;
    margin: 0 !important; padding: 0 !important; gap: 0 !important;
}

.left-car-panel {
    flex: 0 0 30% !important; max-width: 30% !important; height: 100% !important;
    background: #fdfdfd !important;
    margin: 0 !important; padding: 40px 20px !important;
    border-right: 1px solid #eee !important; border-radius: 0 !important;
    display: flex !important; flex-direction: column !important;
    justify-content: space-around !important; align-items: center !important;
}

.speed-display { text-align: center; color: #222;}
.speed-num { font-size: 85px; font-weight: 600; line-height: 1; }
.speed-unit { font-size: 14px; color: #999; font-weight: bold; letter-spacing: 2px; }
.gear-selector { font-size: 18px; color: #ccc; font-weight: bold; letter-spacing: 8px; }
.gear-active { color: #111; font-size: 20px; font-weight: 900;}
.car-mockup { font-size: 140px; line-height: 1; text-shadow: 0 15px 25px rgba(0,0,0,0.15); margin-top: -20px;}
.battery-container { width: 80%; display: flex; flex-direction: column; align-items: center; gap: 8px;}

.right-content-panel {
    flex: 1 !important; height: 100% !important;
    background: #111111 !important;
    margin: 0 !important; padding: 30px 50px 50px 50px !important;
    border-radius: 0 !important; overflow-y: auto !important; scrollbar-width: none;
}
.right-content-panel::-webkit-scrollbar { display: none; }
.clear-bg { background: transparent !important; border: none !important; box-shadow: none !important; }

.alert-box { padding: 25px 20px; border-radius: 12px; text-align: center; margin-bottom: 15px; color: #fff;}
.alert-title { font-size: 22px; font-weight: 600; margin-bottom: 6px; letter-spacing: 1px;}
.alert-subtitle { font-size: 13px; opacity: 0.7; letter-spacing: 1px;}
.safe { background: rgba(46,204,113,0.08); color: #2ecc71; border: 1px solid rgba(46,204,113,0.2); }
.warning { background: rgba(241,196,15,0.08); color: #f1c40f; border: 1px solid rgba(241,196,15,0.2); }
.danger { background: rgba(231,76,60,0.12); color: #e74c3c; border: 1px solid rgba(231,76,60,0.4); animation: flash 1.2s infinite alternate; }
@keyframes flash { from { box-shadow: 0 0 10px rgba(231,76,60,0.1); } to { box-shadow: 0 0 40px rgba(231,76,60,0.5); } }

.bottom-nav-bar {
    position: fixed !important; bottom: 0 !important; left: 0 !important;
    width: 100vw !important; height: 80px !important;
    background: #000 !important; color: #fff;
    display: flex !important; justify-content: space-around !important; align-items: center !important;
    font-size: 26px; border-top: 1px solid #1a1a1a !important; z-index: 9999 !important;
}
.nav-temp { font-size: 20px; font-weight: 600; letter-spacing: 1px;}
"""

with gr.Blocks(title="HyperV-DMS") as demo:

    gr.HTML("""
        <div class="top-status-bar">
            <div><span style="opacity:0.5">🔒</span> &nbsp;&nbsp; 📱 &nbsp;&nbsp; 📶 5G</div>
            <div style="font-weight: 900; letter-spacing: 2px;">HYPERV-DMS</div>
            <div>22°C &nbsp;&nbsp; 10:21 AM</div>
        </div>
    """)

    with gr.Row(elem_classes="middle-wrapper"):

        with gr.Column(elem_classes="left-car-panel"):
            gr.HTML("""
                <div class="speed-display">
                    <div class="speed-num">63</div>
                    <div class="speed-unit">MPH</div>
                </div>
                <div class="gear-selector">
                    P &nbsp; R &nbsp; N &nbsp; <span class="gear-active">D</span>
                </div>
                <div class="car-mockup">🚘</div>
                <div class="battery-container">
                    <div style="width: 100%; height: 6px; background: #eee; border-radius: 3px; overflow: hidden;">
                        <div style="width: 90%; height: 100%; background: #2ecc71;"></div>
                    </div>
                    <div style="font-size: 13px; color: #888; font-weight: bold; width: 100%; text-align: left;">🔋 90%</div>
                </div>
            """)

        with gr.Column(elem_classes="right-content-panel"):

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("<h3 style='color: #eee; margin: 0 0 10px 0; font-size: 13px; letter-spacing: 1px;'>📸 CABIN CAMERA (支持本地录像上传)</h3>")
                    video_input = gr.Video(
                        sources=["webcam", "upload"],
                        height=240
                    )

                with gr.Column(scale=1):
                    gr.Markdown("<h3 style='color: #eee; margin: 0 0 10px 0; font-size: 13px; letter-spacing: 1px;'>🎙️ CABIN MIC (声纹阵列)</h3>")
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath"
                    )

            gr.Markdown("<h3 style='color: #eee; margin: 20px 0 10px 0; font-size: 13px; letter-spacing: 1px;'>🧠 NEURAL TELEMETRY (双模态融合分析)</h3>")

            status_output = gr.HTML('<div class="alert-box safe"><div class="alert-title">系统就绪</div><div class="alert-subtitle">请开启摄像头，或将测试视频拖拽至左侧虚线框内</div></div>')
            label_output = gr.Label(label="", num_top_classes=5, elem_classes="clear-bg")

            btn = gr.Button("激活多模态舱内分析", variant="primary", elem_classes="clear-bg")

    gr.HTML("""
        <div class="bottom-nav-bar">
            <div style="cursor: pointer; opacity: 0.8;">🚗</div>
            <div style="cursor: pointer; opacity: 0.8;">❄️</div>
            <div style="cursor: pointer; opacity: 0.8;">♨️</div>
            <div class="nav-temp">20°</div>
            <div style="cursor: pointer; opacity: 0.8;">🎵</div>
            <div style="cursor: pointer; opacity: 0.8;">📞</div>
            <div style="cursor: pointer; opacity: 0.8;">🔊</div>
        </div>
    """)

    btn.click(fn=dms_inference, inputs=[video_input, audio_input], outputs=[label_output, status_output])

if __name__ == "__main__":
    demo.launch(share=True, debug=True, css=car_css)
