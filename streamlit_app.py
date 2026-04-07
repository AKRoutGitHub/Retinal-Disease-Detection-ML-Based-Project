import io
from typing import Dict, List

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import streamlit.components.v1 as components
import torchvision.models as models
import torchvision.transforms as transforms


# ---------- Page Setup ----------
st.set_page_config(
    page_title="JyotirMaya",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Styling ----------
st.markdown(
    """
    <style>
    .main {background: linear-gradient(180deg, #08111f 0%, #0e1a2f 100%);}
    .stApp {color: #eaf3ff;}
    .title-card {
        background: linear-gradient(120deg, #1f6feb 0%, #8250df 100%);
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .game-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 14px;
        padding: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.6rem;
        border-radius: 999px;
        background: #1a7f37;
        color: white;
        font-size: 0.82rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .glow-card {
        border: 1px solid rgba(250, 204, 21, 0.75);
        box-shadow: 0 0 14px rgba(250, 204, 21, 0.45), 0 0 36px rgba(245, 158, 11, 0.25);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        background: rgba(250, 204, 21, 0.08);
    }
    .who-link-box {
        display: block;
        text-decoration: none;
        color: #eaf3ff;
        border: 1px solid rgba(96, 165, 250, 0.65);
        border-radius: 12px;
        background: rgba(59, 130, 246, 0.16);
        padding: 0.9rem;
        min-height: 108px;
        transition: all 0.2s ease;
    }
    .who-link-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 14px rgba(96, 165, 250, 0.5);
        border-color: rgba(59, 130, 246, 0.95);
    }
    .who-link-title {
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .who-link-sub {
        font-size: 0.84rem;
        color: #c6d7f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Exact Transform Pipeline ----------
preprocess_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def prepare_image_for_model(image_bytes: bytes, device: torch.device) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed_image = preprocess_transform(image)
    processed_image = processed_image.unsqueeze(0).to(device)  # (B, C, H, W)
    return processed_image


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, in_planes, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_vec = self.avg_pool(x).view(b, c)
        max_vec = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg_vec).view(b, c, 1, 1)
        max_out = self.fc(max_vec).view(b, c, 1, 1)
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class EfficientNetCBAM(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()  # checkpoint only stores custom classifier
        self.cbam = CBAM(in_planes=1280, ratio=16)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.cbam(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@st.cache_resource
def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict) or "classifier.weight" not in state:
        raise ValueError("Unsupported checkpoint format. Expected a state_dict with 'classifier.weight'.")

    num_classes = int(state["classifier.weight"].shape[0])
    model = EfficientNetCBAM(num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def predict(
    model: nn.Module,
    image_bytes: bytes,
    device: torch.device,
    threshold: float,
    class_names: List[str],
) -> Dict[str, float]:
    input_tensor = prepare_image_for_model(image_bytes, device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.sigmoid(output).squeeze(0).cpu().numpy()
    return {name: float(prob) for name, prob in zip(class_names, probabilities)}


def risk_label(score: float) -> str:
    if score >= 0.8:
        return "Critical Alert"
    if score >= 0.5:
        return "Needs Review"
    if score >= 0.2:
        return "Low-Mid Risk"
    return "Likely Normal"


def get_who_link(label: str) -> str:
    lower = label.lower()
    if "normal" in lower:
        return "https://www.who.int/health-topics/blindness-and-vision-loss"
    if "diabetes" in lower:
        return "https://www.who.int/news-room/fact-sheets/detail/diabetes"
    if "glaucoma" in lower:
        return "https://www.who.int/health-topics/blindness-and-vision-loss"
    if "cataract" in lower:
        return "https://www.who.int/health-topics/blindness-and-vision-loss"
    if "macular" in lower or "amd" in lower:
        return "https://www.who.int/health-topics/blindness-and-vision-loss"
    if "hypertension" in lower:
        return "https://www.who.int/news-room/fact-sheets/detail/hypertension"
    if "myopia" in lower:
        return "https://www.who.int/health-topics/blindness-and-vision-loss"
    return "https://www.who.int/health-topics/blindness-and-vision-loss"


st.markdown(
    """
    <div class="title-card">
      <h1 style="margin:0;">👁️ JyotirMaya</h1>
      <p style="margin:0.3rem 0 0 0;">Upload fundus image → preprocess → infer → view gamified results dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("⚙️ Model Controls")
    default_checkpoint = "best_model.pth"
    checkpoint_path = st.text_input("Model checkpoint path", value=default_checkpoint)
    class_text = st.text_area(
        "Disease labels (comma separated)",
        value="Normal (N), Diabetes (D), Glaucoma (G), Cataract (C), Age related Macular Degeneration (A), Hypertension (H), Pathological Myopia (M), Other diseases/abnormalities (O)",
    )
    threshold = st.slider("Prediction threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    if "uploaded_file" in locals() and uploaded_file is not None:
        st.markdown(
            """
            <style>
            section[data-testid="stSidebar"] div[data-testid="stButton"] button {
                border: 1px solid #facc15 !important;
                box-shadow: 0 0 10px rgba(250, 204, 21, 0.6), 0 0 26px rgba(245, 158, 11, 0.35) !important;
                animation: pulseGlow 1.2s ease-in-out infinite alternate;
            }
            @keyframes pulseGlow {
                from { box-shadow: 0 0 8px rgba(250, 204, 21, 0.45), 0 0 16px rgba(245, 158, 11, 0.25); }
                to { box-shadow: 0 0 14px rgba(250, 204, 21, 0.75), 0 0 30px rgba(245, 158, 11, 0.45); }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    run_button = st.button("🚀 Analyze Image", use_container_width=True)


class_names = [x.strip() for x in class_text.split(",") if x.strip()]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


left_col, right_col = st.columns([1.05, 1.2], gap="large")
analysis_results: Dict[str, float] | None = None

with left_col:
    st.subheader("📤 Upload Fundus Image")
    uploaded_file = st.file_uploader(
        "Choose image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Best results with clear retinal fundus images.",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Input Image", use_container_width=True)
        st.sidebar.markdown(
            """
            <style>
            section[data-testid="stSidebar"] div[data-testid="stButton"] button {
                border: 1px solid #facc15 !important;
                box-shadow: 0 0 10px rgba(250, 204, 21, 0.6), 0 0 26px rgba(245, 158, 11, 0.35) !important;
                animation: pulseGlow 1.2s ease-in-out infinite alternate;
            }
            @keyframes pulseGlow {
                from { box-shadow: 0 0 8px rgba(250, 204, 21, 0.45), 0 0 16px rgba(245, 158, 11, 0.25); }
                to { box-shadow: 0 0 14px rgba(250, 204, 21, 0.75), 0 0 30px rgba(245, 158, 11, 0.45); }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

with right_col:
    st.subheader("🎮 Diagnostic Arena")
    st.markdown('<div class="game-card">', unsafe_allow_html=True)
    st.write(f"**Device:** `{device}`")
    st.write("Press **Analyze Image** to start the quest and unlock disease risk badges.")

    if run_button:
        if uploaded_file is None:
            st.warning("Upload an image first.")
        else:
            try:
                model = load_model(checkpoint_path, device)
                model_num_classes = model.classifier.out_features

                if len(class_names) != model_num_classes:
                    class_names = [f"Class {i + 1}" for i in range(model_num_classes)]
                    st.info(
                        "Label count did not match checkpoint output classes. "
                        f"Using auto labels for {model_num_classes} classes."
                    )

                scores = predict(
                    model=model,
                    image_bytes=uploaded_file.read(),
                    device=device,
                    threshold=threshold,
                    class_names=class_names,
                )
                analysis_results = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
                st.success("Inference complete. Scroll below for full-screen dashboard results.")

            except Exception as exc:
                st.error(
                    "Could not run inference. Verify model path/class architecture/checkpoint format.\n\n"
                    f"Error: {exc}"
                )

    st.markdown("</div>", unsafe_allow_html=True)


if analysis_results is not None:
    st.write("")
    st.markdown("## 📊 Full-Screen Prediction Dashboard")
    sorted_items = list(analysis_results.items())
    top3_labels = [name for name, _ in sorted_items[:3]]
    warning_palette = ["#ef4444", "#f59e0b", "#facc15"]  # red, orange, yellow
    color_map: Dict[str, str] = {}
    top3_non_normal_rank = 0
    for name, _ in sorted_items:
        if "normal" in name.lower():
            color_map[name] = "#22c55e"
        elif name in top3_labels and top3_non_normal_rank < len(warning_palette):
            color_map[name] = warning_palette[top3_non_normal_rank]
            top3_non_normal_rank += 1
        else:
            color_map[name] = "#60a5fa"

    chart_df = pd.DataFrame(
        {
            "Disease": [name for name, _ in sorted_items],
            "Probability": [score for _, score in sorted_items],
            "Color": [color_map[name] for name, _ in sorted_items],
        }
    )
    chart_df["Probability %"] = (chart_df["Probability"] * 100).round(2)

    st.vega_lite_chart(
        chart_df,
        {
            "mark": {"type": "bar", "cornerRadiusTopLeft": 6, "cornerRadiusTopRight": 6},
            "encoding": {
                "x": {"field": "Disease", "type": "nominal", "sort": "-y"},
                "y": {"field": "Probability %", "type": "quantitative", "title": "Probability (%)"},
                "color": {"field": "Color", "type": "nominal", "scale": None, "legend": None},
                "tooltip": [
                    {"field": "Disease", "type": "nominal"},
                    {"field": "Probability %", "type": "quantitative"},
                ],
            },
        },
        use_container_width=True,
    )

    detected = [name for name, score in sorted_items if score > threshold]
    if detected:
        st.markdown("###  Detected Disease Badges")
        badges_html = "".join([f'<span class="badge">{name}</span>' for name in detected])
        st.markdown(badges_html, unsafe_allow_html=True)
    else:
        st.info("No disease crossed the selected threshold.")

    st.markdown("<div id='probability-table-anchor'></div>", unsafe_allow_html=True)
    st.markdown("### 📋 Probability Table")
    table_df = chart_df[["Disease", "Probability %"]].copy()
    table_df["Risk Tag"] = [risk_label(v / 100.0) for v in table_df["Probability %"]]
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    components.html(
        """
        <script>
            const anchor = window.parent.document.getElementById("probability-table-anchor");
            if (anchor) {
                anchor.scrollIntoView({behavior: "smooth", block: "start"});
            }
        </script>
        """,
        height=0,
    )

    st.markdown("### 📚 WHO Documentation (Top 3 Predictions)")
    who_cols = st.columns(3, gap="medium")
    for idx, (label, score) in enumerate(sorted_items[:3], start=1):
        with who_cols[idx - 1]:
            link = get_who_link(label)
            st.markdown(
                f"""
                <a class="who-link-box" href="{link}" target="_blank">
                    <div class="who-link-title">#{idx} {label}</div>
                    <div><strong>{score:.2%}</strong> probability</div>
                    <div class="who-link-sub">Open WHO documentation</div>
                </a>
                """,
                unsafe_allow_html=True,
            )

    top_label, top_score = sorted_items[0]
    st.markdown("### 🧠 Final AI Hint")
    st.markdown(
        f"""
        <div class="glow-card">
            Most likely class: <strong>{top_label}</strong> with confidence <strong>{top_score:.2%}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
