# Retina AI Quest - Fundus Disease Prediction Dashboard

A visually rich, practitioner-friendly Streamlit app for retinal fundus image analysis using a PyTorch `EfficientNet-B0 + CBAM` model.

Upload an eye fundus image, run inference, and review a gamified prediction dashboard with color-prioritized risk visibility and top-3 WHO reference links.

---

## Live Demo (Deployment)


- **App Link:** https://jyotirmaya-giet-vision-g11-asis-krishn-subham.streamlit.app/
- **Status:** `Coming Soon`

---

## Features

- **One-click inference flow**
  - Upload image -> preprocess -> infer -> visualize predictions.
- **Exact test-time preprocessing**
  - `Resize(224,224)` -> `ToTensor()` -> ImageNet normalization.
- **Custom model loading**
  - Loads `best_model.pth` (`state_dict`) into `EfficientNetCBAM`.
- **Gamified dashboard**
  - Badges, colorful full-width chart, probability table, and highlighted final AI hint.
- **Risk-focused color coding**
  - `Normal` class in green.
  - Top non-normal classes in warning colors for rapid attention.
- **Top-3 documentation support**
  - Clickable horizontal WHO resource boxes for highest-probability predictions.

---

## Disease Classes (8)

1. Normal (N)
2. Diabetes (D)
3. Glaucoma (G)
4. Cataract (C)
5. Age related Macular Degeneration (A)
6. Hypertension (H)
7. Pathological Myopia (M)
8. Other diseases/abnormalities (O)

---

## Project Structure

```text
sem_06_project_final/
|-- .gitignore
|-- README.md
|-- best_model.pth
|-- streamlit_app.py
|-- testing_images/           # local sample/test images (ignored in git)
|-- .venv/                    # local virtual environment (ignored in git)
`-- __pycache__/              # Python cache (ignored in git)
```

### File Responsibilities

- `streamlit_app.py`
  - App UI, model architecture definition (`EfficientNet-B0 + CBAM`), checkpoint loading, inference logic, and interactive visual dashboard.
- `best_model.pth`
  - Trained model checkpoint used for predictions.
- `.gitignore`
  - Excludes local runtime/developer artifacts (`.venv`, caches, test assets).

---

## Tech Stack

- **Frontend / App:** Streamlit
- **Model Runtime:** PyTorch, TorchVision
- **Image Processing:** Pillow, TorchVision Transforms
- **Data Display:** Pandas + Vega-Lite charts via Streamlit

---

## Local Setup

### 1) Create and activate virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2) Install dependencies

```powershell
python -m pip install --upgrade pip
pip install streamlit torch torchvision pillow pandas
```

### 3) Run the app

```powershell
streamlit run streamlit_app.py
```

---

## Inference Pipeline

1. Load image from upload widget.
2. Convert to RGB.
3. Apply preprocessing transform:
   - Resize to `224 x 224`
   - Convert to tensor
   - Normalize with ImageNet mean/std
4. Add batch dimension and move tensor to selected device (CPU/GPU).
5. Run model in eval mode with `torch.no_grad()`.
6. Apply `sigmoid` for multilabel probabilities.
7. Render:
   - Sorted probabilities
   - Color-priority chart/table
   - Top-3 WHO links
   - Highlighted final prediction

---

## Deployment Notes

When deploying (Streamlit Community Cloud, Render, Hugging Face Spaces, etc.):

- Keep `best_model.pth` accessible in app root or update path in sidebar.
- Ensure Python version and torch/torchvision compatibility.
- Add a `requirements.txt` if your platform needs explicit dependency resolution.

### Deployment Checklist

- [ ] App runs locally from `streamlit run streamlit_app.py`
- [ ] Model checkpoint loads without mismatch
- [ ] Upload + inference works
- [ ] Top-3 WHO links open correctly
- [ ] Add deployed URL in **Live Demo (Deployment)** section above

---

## Disclaimer

This project is intended for research/educational assistance and decision support.  
It is **not** a substitute for licensed medical diagnosis or treatment.

