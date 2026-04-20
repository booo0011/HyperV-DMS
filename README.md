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
1. Create a Python 3.12 virtual environment and activate it:
   ```bash
   py -3.12 -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (required for audio processing):
   - **Windows**: Run `install_ffmpeg_windows.bat` as administrator, or manually download from https://ffmpeg.org/download.html and add to PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

3. Place your trained weights in `models/resume_voyagernet_epoch50.pth`.

4. Run the app:
   ```bash
   python app.py
   ```

## Notebook
The original notebook has been relocated to `notebooks/model_training_and_eda.ipynb`.
