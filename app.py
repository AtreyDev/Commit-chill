"""
VeriLogic — AI Document Forgery Detection and Analysis
Streamlit multi-tab application.

Run with:
    streamlit run app.py

Tabs:
  1. Signature Verification    — Siamese network pair comparison
  2. Copy-Move Detection       — ORB+RANSAC / photoholmes
  3. Document Analysis         — ELA, edge detection, OCR, wavelet
  4. AI Screening              — unified explainable risk report
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image, UnidentifiedImageError
from api.index import handler

try:
    from streamlit_image_comparison import image_comparison
    HAS_IMAGE_COMPARISON = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_IMAGE_COMPARISON = False

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VeriLogic — AI Document Screening",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

VERDICT_STYLES = {
    "Authentic": {"color": "#22C55E", "bg": "rgba(34, 197, 94, 0.14)", "icon": "✅"},
    "Genuine":   {"color": "#22C55E", "bg": "rgba(34, 197, 94, 0.14)", "icon": "✅"},
    "Suspicious": {"color": "#F59E0B", "bg": "rgba(245, 158, 11, 0.14)", "icon": "⚠️"},
    "Forged":    {"color": "#EF4444", "bg": "rgba(239, 68, 68, 0.14)", "icon": "🚫"},
}
DEFAULT_VERDICT_STYLE = {"color": "#9CA3AF", "bg": "rgba(156, 163, 175, 0.14)", "icon": "❔"}


# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  { font-family: 'Inter', -apple-system, sans-serif; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px; }

        /* ── Hero ─────────────────────────────────────────────────────── */
        .cc-hero {
            background: linear-gradient(135deg, rgba(79,142,247,0.20) 0%, rgba(124,58,237,0.14) 100%);
            border: 1px solid rgba(79,142,247,0.25);
            border-radius: 20px;
            padding: 28px 32px;
            margin-bottom: 22px;
        }
        .cc-hero h1 { font-size: 1.9rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.02em; }
        .cc-hero p { margin: 0; opacity: 0.82; font-size: 0.98rem; }
        .cc-chip-row { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
        .cc-chip {
            display: inline-block; padding: 4px 12px; border-radius: 999px;
            background: rgba(250,250,250,0.08); border: 1px solid rgba(250,250,250,0.14);
            font-size: 0.76rem; font-weight: 500; opacity: 0.9;
        }

        /* ── Section headers ─────────────────────────────────────────── */
        .cc-section-title { font-size: 1.18rem; font-weight: 700; margin: 4px 0 2px 0; }
        .cc-section-sub { opacity: 0.68; font-size: 0.92rem; margin-bottom: 14px; }

        /* ── Verdict pill ─────────────────────────────────────────────── */
        .cc-verdict {
            display: flex; align-items: center; gap: 12px;
            padding: 14px 20px; border-radius: 14px; margin: 6px 0 14px 0;
            background: var(--vbg); border: 1px solid var(--vc);
        }
        .cc-verdict .icon { font-size: 1.6rem; line-height: 1; }
        .cc-verdict .text { font-size: 1.25rem; font-weight: 800; color: var(--vc); }
        .cc-verdict .caption { font-size: 0.85rem; opacity: 0.75; margin-top: 2px; }

        /* ── Empty state ──────────────────────────────────────────────── */
        .cc-empty {
            border: 1.5px dashed rgba(250,250,250,0.22); border-radius: 16px;
            padding: 34px 20px; text-align: center; opacity: 0.85;
        }
        .cc-empty .icon { font-size: 2rem; margin-bottom: 6px; }
        .cc-empty .title { font-weight: 600; font-size: 0.98rem; margin-bottom: 4px; }
        .cc-empty .sub { font-size: 0.84rem; opacity: 0.65; }

        /* ── Badge chips (wellness) ───────────────────────────────────── */
        .cc-badge-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 6px; }
        .cc-badge {
            background: rgba(79,142,247,0.12); border: 1px solid rgba(79,142,247,0.3);
            border-radius: 12px; padding: 10px 14px; min-width: 160px;
        }
        .cc-badge .name { font-weight: 700; font-size: 0.92rem; }
        .cc-badge .desc { font-size: 0.78rem; opacity: 0.72; margin-top: 2px; }

        /* ── Notification card ────────────────────────────────────────── */
        .cc-notif {
            border-left: 3px solid #4F8EF7; background: rgba(79,142,247,0.08);
            border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
        }
        .cc-notif b { font-size: 0.92rem; }
        .cc-notif .msg { font-size: 0.85rem; opacity: 0.78; }

        /* ── Streamlit widget polish ──────────────────────────────────── */
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 10px; font-weight: 600; padding: 0.55rem 1.1rem;
            transition: transform 0.06s ease-in-out;
        }
        div.stButton > button[kind="primary"]:hover { transform: translateY(-1px); }

        [data-testid="stFileUploaderDropzone"] {
            border-radius: 14px !important; border: 1.5px dashed rgba(79,142,247,0.35) !important;
        }

        [data-testid="stMetric"] {
            background: rgba(250,250,250,0.03); border: 1px solid rgba(250,250,250,0.08);
            border-radius: 12px; padding: 12px 16px 8px 16px;
        }

        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0; padding: 8px 16px; font-weight: 600;
        }

        [data-testid="stExpander"] { border-radius: 12px; border: 1px solid rgba(250,250,250,0.08); }

        [data-testid="stSidebar"] { border-right: 1px solid rgba(250,250,250,0.08); }
        .cc-side-step {
            display: flex; gap: 10px; align-items: flex-start; margin-bottom: 12px;
        }
        .cc-side-step .num {
            flex-shrink: 0; width: 22px; height: 22px; border-radius: 50%;
            background: rgba(79,142,247,0.22); color: #4F8EF7; font-weight: 700;
            font-size: 0.78rem; display: flex; align-items: center; justify-content: center;
        }
        .cc-side-step .label { font-size: 0.85rem; opacity: 0.85; padding-top: 2px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reusable UI components
# ─────────────────────────────────────────────────────────────────────────────

def hero() -> None:
    st.markdown(
        """
        <div class="cc-hero">
            <h1>🔍 VeriLogic — AI Document Screening</h1>
            <p>Explainable OCR, MRZ validation, and image-forensics evidence for officer review.</p>
            <div class="cc-chip-row">
                <span class="cc-chip">🧠 Siamese networks</span>
                <span class="cc-chip">🧩 ORB + RANSAC</span>
                <span class="cc-chip">🩻 Error Level Analysis</span>
                <span class="cc-chip">🔤 EasyOCR</span>
                <span class="cc-chip">🌊 PyWavelets</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(icon: str, title: str, subtitle: str | None = None) -> None:
    sub_html = f'<p class="cc-section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<p class="cc-section-title">{icon} {title}</p>{sub_html}',
        unsafe_allow_html=True,
    )


def verdict_badge(verdict: str, caption: str | None = None) -> None:
    style = VERDICT_STYLES.get(verdict, DEFAULT_VERDICT_STYLE)
    caption_html = f'<div class="caption">{caption}</div>' if caption else ""
    st.markdown(
        f"""
        <div class="cc-verdict" style="--vc:{style['color']}; --vbg:{style['bg']};">
            <span class="icon">{style['icon']}</span>
            <div>
                <div class="text">{verdict}</div>
                {caption_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Kept for any external/legacy callers that still expect the original name.
def _verdict_badge(verdict: str) -> None:
    verdict_badge(verdict)


def empty_state(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="cc-empty">
            <div class="icon">{icon}</div>
            <div class="title">{title}</div>
            <div class="sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _as_pil(img) -> Image.Image:
    """Best-effort conversion of a PIL Image / numpy array to an RGB PIL Image."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def compare_images(img1, img2, label1: str, label2: str) -> None:
    """Interactive before/after slider when available, side-by-side otherwise."""
    if HAS_IMAGE_COMPARISON:
        try:
            image_comparison(
                img1=_as_pil(img1),
                img2=_as_pil(img2),
                label1=label1,
                label2=label2,
                width=640,
                make_responsive=True,
            )
            return
        except Exception:
            pass  # fall back to plain columns below
    c1, c2 = st.columns(2)
    c1.image(img1, caption=label1, use_container_width=True)
    c2.image(img2, caption=label2, use_container_width=True)


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🔍 VeriLogic")
        st.caption("AI-assisted document forensics toolkit")
        st.divider()
        st.markdown("**How to use this tool**")
        steps = [
            "Pick the tab that matches your task below",
            "Upload a document or signature image (≤ 10 MB)",
            "Run the analysis and review the verdict",
            "Export a JSON/PDF report for the audit trail",
        ]
        for i, step in enumerate(steps, start=1):
            st.markdown(
                f'<div class="cc-side-step"><div class="num">{i}</div>'
                f'<div class="label">{step}</div></div>',
                unsafe_allow_html=True,
            )
        st.divider()
        st.markdown("**Tabs**")
        st.markdown(
            "- 🌿 Optional wellness\n"
            "- ✍️ Signature Verification\n"
            "- 🔍 Copy-Move Detection\n"
            "- 📄 Document Analysis\n"
            "- 🛡️ AI Screening"
        )
        st.divider()
        st.caption("Uploads are validated, size-capped, and processed locally for this session.")
        st.caption("Built with PyTorch · OpenCV · EasyOCR · PyWavelets · Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_upload(uploaded) -> Path:
    """Validate and save an uploaded image without trusting its filename."""
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError("Unsupported image type")
    content = uploaded.getvalue()
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 10 MB upload limit")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a valid image") from exc
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)


inject_css()
sidebar()
hero()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dashboard, tab1, tab2, tab3, tab4 = st.tabs([
    "🌿  Optional wellness",
    "✍️  Signature Verification",
    "🔍  Copy-Move Detection",
    "📄  Document Analysis",
    "🛡️  AI Screening",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — Secondary wellness dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab_dashboard:
    from src.gamification import calculate_points, get_badges
    from src.github_client import GitHubClient, GitHubClientError, demo_activities
    from src.notifications import build_chill_notifications
    from src.productivity import (
        aggregate_activity,
        calculate_burnout_signal,
        chart_rows,
        filter_activity,
        healthy_streak,
    )

    section_header(
        "🌿", "Wellness dashboard",
        "Track productivity while protecting work-life balance. This dashboard uses activity "
        "patterns, not medical diagnosis, to suggest healthier ways to work.",
    )

    with st.container(border=True):
        mode_col, user_col, days_col = st.columns([1, 2, 1])
        with mode_col:
            demo_mode = st.toggle("Demo mode", value=True, help="Use synthetic activity for a presentation-safe demo.")
        with user_col:
            github_user = st.text_input("GitHub username", value="atreydev", disabled=demo_mode)
        with days_col:
            window_days = st.selectbox("Window", [7, 14, 30], index=2)

        token = st.secrets.get("GITHUB_TOKEN") if not demo_mode else None
        if not demo_mode and not token:
            st.info("Add GITHUB_TOKEN to Streamlit secrets for GitHub API mode. Demo mode stays fully offline.")

    if demo_mode:
        raw_activities = demo_activities()
    elif github_user and token:
        try:
            raw_activities = GitHubClient(token).fetch_user_events(github_user)
        except (GitHubClientError, ValueError) as error:
            st.error(str(error))
            raw_activities = []
    else:
        raw_activities = []

    activities = filter_activity(raw_activities, days=window_days)
    metrics = aggregate_activity(activities)
    burnout = calculate_burnout_signal(metrics)
    if "breaks_taken" not in st.session_state:
        st.session_state.breaks_taken = 0
    if "dismissed_chill" not in st.session_state:
        st.session_state.dismissed_chill = set()
    points = calculate_points(metrics, breaks_taken=st.session_state.breaks_taken)
    badges = get_badges(metrics, breaks_taken=st.session_state.breaks_taken)
    notifications = [
        item for item in build_chill_notifications(metrics, burnout)
        if item["id"] not in st.session_state.dismissed_chill
    ]

    st.caption("Demo data is synthetic. GitHub activity is an imperfect proxy for effort and wellbeing.")
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("📊 Activity", metrics["total"])
    metric2.metric("🔥 Healthy streak", f"{healthy_streak(activities)} days")
    metric3.metric("🍃 Chill points", points)
    metric4.metric("⚖️ Workload signal", f"{burnout['level']} · {burnout['score']:.0%}")

    chart_data = chart_rows(metrics)
    if chart_data:
        fig = px.bar(chart_data, x="date", y="activity", title="Activity by day")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=10, l=10, r=10), font=dict(family="Inter"),
        )
        fig.update_traces(marker_color="#4F8EF7")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Why this signal?")
            st.progress(burnout["score"], text=f"{burnout['level']} workload signal")
            for name, value in burnout["factors"].items():
                st.write(f"{name.replace('_', ' ').title()}: {value:.0%}")
            st.caption(burnout["disclaimer"])
    with right:
        with st.container(border=True):
            st.subheader("Chill mode")
            if st.button("☕ Take a 5-minute break", key="take_break", type="primary", use_container_width=True):
                st.session_state.breaks_taken += 1
                st.success("Break logged. Step away from the screen for five minutes.")
            st.write("Try one now:")
            st.write("• Look away from the screen and relax your shoulders")
            st.write("• Get water or take a short walk")
            st.write("• Set a stop-work time before your next task")

    st.subheader("Chill notifications")
    if not notifications:
        st.success("No urgent nudges. Keep your pace sustainable.")
    for notification in notifications:
        notification_col, action_col = st.columns([5, 1])
        with notification_col:
            st.markdown(
                f"""<div class="cc-notif"><b>{notification['title']}</b>
                <div class="msg">{notification['message']}</div></div>""",
                unsafe_allow_html=True,
            )
        if action_col.button("Dismiss", key=f"dismiss_{notification['id']}"):
            st.session_state.dismissed_chill.add(notification["id"])
            st.rerun()

    st.subheader("Personal badges")
    if badges:
        chips = "".join(
            f'<div class="cc-badge"><div class="name">🏅 {b["name"]}</div>'
            f'<div class="desc">{b["description"]}</div></div>'
            for b in badges
        )
        st.markdown(f'<div class="cc-badge-grid">{chips}</div>', unsafe_allow_html=True)
    else:
        empty_state("🏅", "No badges yet", "Your first badge is waiting for a healthy-work habit.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Signature Verification
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    section_header(
        "✍️", "Offline Signature Verification",
        "Upload a **reference** (enrolled) signature and a **query** (candidate) signature. "
        "The Siamese network compares their embeddings and determines if they match.",
    )

    col_ref, col_qry = st.columns(2)
    with col_ref:
        with st.container(border=True):
            st.markdown("**Reference signature**")
            ref_file = st.file_uploader(
                "Reference signature", type=["png", "jpg", "jpeg"], key="sig_ref",
                label_visibility="collapsed",
            )
            if ref_file:
                st.image(ref_file, caption="Reference", use_container_width=True)
            else:
                empty_state("📥", "No reference uploaded", "The enrolled / known-genuine signature")

    with col_qry:
        with st.container(border=True):
            st.markdown("**Query signature**")
            qry_file = st.file_uploader(
                "Query signature", type=["png", "jpg", "jpeg"], key="sig_qry",
                label_visibility="collapsed",
            )
            if qry_file:
                st.image(qry_file, caption="Query", use_container_width=True)
            else:
                empty_state("📥", "No query uploaded", "The candidate signature to verify")

    with st.expander("⚙️  Advanced — model weights"):
        weights_path = st.text_input(
            "Model weights path", value="weights/siamese_best.pt",
            help="Run `python -m src.signature.train` to generate weights."
        )

    st.button(
        "🔎 Verify Signatures", key="verify_sig_btn", type="primary",
        disabled=not (ref_file and qry_file), use_container_width=True,
    )

    if st.session_state.get("verify_sig_btn"):
        ref_path = _save_upload(ref_file)
        qry_path = _save_upload(qry_file)
        weights = Path(weights_path)

        if not weights.exists():
            st.warning(
                f"Weights file `{weights_path}` not found. "
                "Train the model first with:\n"
                "```\npython -m src.signature.train\n```"
            )
        else:
            with st.spinner("Running Siamese network..."):
                from src.signature.inference import verify
                result = verify(ref_path, qry_path, weights=weights)

            verdict_badge(result["verdict"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{result['confidence']:.1%}")
            m2.metric("Cosine Distance", f"{result['distance']:.4f}")
            m3.metric("Match", "Yes ✓" if result["match"] else "No ✗")

    st.divider()
    with st.expander("ℹ️  About the model"):
        st.markdown("""
**Architecture**: Siamese Network with shared EfficientNet-B0 backbone (timm) +
projection head (Linear → BN → ReLU → Dropout → Linear).

**Training**: Contrastive loss (pytorch-metric-learning), AdamW optimiser,
CosineAnnealingLR scheduler. Default 30 epochs on CEDAR-style paired data.

**References**:
- HTCSigNet (Pattern Recognition, 2025) — Hybrid Transformer-Conv signature network
- Multi-Scale CNN-CrossViT (Complex & Intelligent Systems, 2025) — 98.85% on CEDAR
- TransOSV (Pattern Recognition, 2023) — First ViT-based writer-independent verification
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Copy-Move Detection
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    section_header(
        "🔍", "Copy-Move Forgery Detection",
        "Upload a document image. The detector identifies regions that have been "
        "copied and pasted within the same image using ORB keypoint matching and "
        "RANSAC geometric verification.",
    )

    img_file = st.file_uploader(
        "Document image", type=["png", "jpg", "jpeg", "tiff"], key="cm_img"
    )

    if img_file:
        img_pil = Image.open(img_file).convert("RGB")
        with st.container(border=True):
            st.image(img_pil, caption="Uploaded image", use_container_width=True)

        st.button("🔎 Detect Copy-Move", key="detect_cm_btn", type="primary", use_container_width=True)

        if st.session_state.get("detect_cm_btn"):
            img_path = _save_upload(img_file)
            with st.spinner("Running copy-move detector..."):
                from src.copy_move.detector import detect_copy_move
                from src.copy_move.visualizer import overlay_heatmap, annotate_regions

                result = detect_copy_move(img_path)

            verdict_badge(result["verdict"])
            m1, m2 = st.columns(2)
            m1.metric("Forgery Score", f"{result['score']:.1%}")
            m2.metric("Detection Method", result["method"])

            section_header("🧪", "Detection results")
            mask = result["mask"]
            if mask.any():
                overlay = overlay_heatmap(np.array(img_pil), mask, alpha=0.4)
                compare_images(img_pil, overlay, "Original", "Heatmap overlay")
            else:
                st.info("No significant copy-move regions detected.")

            if result["heatmap"] is not None:
                st.image(result["heatmap"], caption="Photoholmes heatmap", use_container_width=True)
            elif mask.any():
                annotated = annotate_regions(np.array(img_pil), mask)
                st.image(annotated, caption="Annotated regions", use_container_width=True)
    else:
        empty_state("📄", "No document uploaded", "Upload a PNG, JPG, or TIFF document to scan for copy-move tampering")

    st.divider()
    with st.expander("ℹ️  About the detector"):
        st.markdown("""
**Primary**: [PhotoHolmes](https://github.com/photoholmes/photoholmes) (Splicebuster) when installed.

**Fallback**: ORB feature extraction → BFMatcher → RANSAC homography estimation.
Inlier ratio determines the forgery confidence score.

**References**:
- CMFDFormer (arXiv 2311.13263, 2023): MiT transformer backbone for CMFD
- PhotoHolmes (arXiv 2412.14969, Springer 2025): unified forensics library
- MVSS-Net++ (T-PAMI): multi-view multi-scale supervision
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Document Analysis (ELA + Edge + OCR + Wavelet)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    section_header(
        "📄", "Document Analysis",
        "Upload a document to run Error Level Analysis, edge detection, OCR, and wavelet decomposition.",
    )

    with st.expander("⚙️  Advanced — ELA re-compression quality", expanded=False):
        st.slider(
            "JPEG re-compression quality", min_value=70, max_value=99, value=95,
            help="Lower quality amplifies differences in manipulated regions. Set this before running ELA.",
            key="ela_quality",
        )

    doc_file = st.file_uploader(
        "Document image", type=["png", "jpg", "jpeg", "tiff", "bmp"], key="doc_img"
    )

    if doc_file:
        doc_pil = Image.open(doc_file).convert("RGB")
        with st.container(border=True):
            st.image(doc_pil, caption="Uploaded document", use_container_width=True)

        analysis_options = st.multiselect(
            "Select analyses to run",
            ["Error Level Analysis (ELA)", "Edge Detection", "OCR", "Wavelet Decomposition"],
            default=["Error Level Analysis (ELA)", "Edge Detection"],
        )

        st.button("▶ Run Analysis", key="run_doc_analysis_btn", type="primary", use_container_width=True)

        if st.session_state.get("run_doc_analysis_btn"):
            doc_path = _save_upload(doc_file)

            # ── ELA ───────────────────────────────────────────────────────────
            if "Error Level Analysis (ELA)" in analysis_options:
                section_header("🩻", "Error Level Analysis")
                with st.spinner("Generating ELA map..."):
                    from src.analysis.ela import generate_ela, ela_score
                    ela_quality = st.session_state.get("ela_quality", 95)
                    ela_img = generate_ela(doc_pil, quality=ela_quality, scale=15)
                    score = ela_score(ela_img)

                compare_images(doc_pil, ela_img, "Original", "ELA Map")

                verdict = "Forged" if score > 0.08 else ("Suspicious" if score > 0.03 else "Authentic")
                verdict_badge(verdict)
                st.metric("ELA Intensity Score", f"{score:.4f}")
                st.caption(
                    "Bright regions in the ELA map indicate areas that may have been "
                    "digitally manipulated. Uniform texture suggests an authentic image."
                )

            # ── Edge Detection ─────────────────────────────────────────────────
            if "Edge Detection" in analysis_options:
                section_header("📐", "Edge Detection")
                detector = st.selectbox(
                    "Detector", ["canny", "sobel", "laplacian", "prewitt_x", "prewitt_y"],
                    key="edge_det",
                )
                with st.spinner("Running edge detection..."):
                    from src.analysis.edge_detection import detect_all
                    edges = detect_all(doc_pil)

                compare_images(doc_pil, edges[detector], "Original", f"{detector.capitalize()} edges")

            # ── OCR ───────────────────────────────────────────────────────────
            if "OCR" in analysis_options:
                section_header("🔤", "Optical Character Recognition")
                handwritten = st.toggle("Handwritten text mode (uses TrOCR)", value=False, key="ocr_hw")
                with st.spinner("Extracting text..."):
                    from src.analysis.ocr import extract_text
                    ocr_result = extract_text(doc_path, handwritten=handwritten)

                st.text_area("Extracted text", ocr_result["full_text"], height=200)
                m1, m2 = st.columns(2)
                m1.metric("Avg. Confidence", f"{ocr_result['avg_confidence']:.1%}")
                m2.metric("Engine", ocr_result["engine"])

                if ocr_result["words"]:
                    with st.expander("Word-level results"):
                        import pandas as pd
                        df = pd.DataFrame([
                            {"Text": w["text"], "Confidence": f"{w['confidence']:.1%}"}
                            for w in ocr_result["words"]
                        ])
                        st.dataframe(df, use_container_width=True, hide_index=True)

            # ── Wavelet ────────────────────────────────────────────────────────
            if "Wavelet Decomposition" in analysis_options:
                section_header("🌊", "Wavelet Decomposition")
                col_w, col_l = st.columns(2)
                wavelet = col_w.selectbox("Wavelet", ["haar", "db1", "db4", "sym4"], key="wav_name")
                level = col_l.slider("Decomposition level", 1, 6, 3, key="wav_level")

                with st.spinner("Running wavelet decomposition..."):
                    from src.analysis.wavelet import decompose
                    wav_result = decompose(doc_pil, wavelet=wavelet, level=level)

                compare_images(
                    doc_pil, wav_result["heatmap"],
                    "Original", f"{wavelet} detail heatmap (level {level})",
                )
    else:
        empty_state("📄", "No document uploaded", "Upload an image to run ELA, edge detection, OCR, or wavelet analysis")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — AI-Based Document Screening
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    section_header(
        "🛡️", "AI-Based Fake Identity & Document Screening",
        "Combine visual tamper checks, copy-move detection, OCR confidence, "
        "metadata indicators, and cross-document identity checks into one explainable report.",
    )

    with st.container(border=True):
        document_type = st.selectbox(
            "Document type",
            ["Unknown", "Passport", "Identity card", "Driver license", "Other"],
            key="screen_document_type",
        )
        up1, up2, up3 = st.columns(3)
        with up1:
            primary_file = st.file_uploader(
                "Primary identity document", type=["png", "jpg", "jpeg", "tiff", "bmp"], key="screen_primary"
            )
        with up2:
            second_file = st.file_uploader(
                "Second identity proof (optional)", type=["png", "jpg", "jpeg", "tiff", "bmp"], key="screen_second"
            )
        with up3:
            video_file = st.file_uploader(
                "Document video (optional)",
                type=["mp4", "mov", "avi"],
                key="screen_video",
                help="Frame extraction not configured yet.",
            )
        enable_ocr = st.checkbox("Run EasyOCR for identity fields", value=False, key="screen_ocr")
        st.info(
            "Face verification is not enabled until a consented, trained face-embedding checkpoint is configured."
        )

    st.button(
        "🛡️ Run Document Screening", key="run_screen_btn", type="primary",
        disabled=not primary_file, use_container_width=True,
    )

    if st.session_state.get("run_screen_btn"):
        from src.analysis.mrz import parse_mrz
        from src.audit import append_audit_record
        from src.export import export_json, export_pdf
        from src.inference.pipeline import build_inference_report
        from src.logic import compare_field_sets
        from src.screening import (
            calculate_risk,
            compare_identity_fields,
            extract_identity_fields,
            screen_document,
        )

        primary_path = _save_upload(primary_file)
        second_path = _save_upload(second_file) if second_file else None
        ocr_results = []
        with st.spinner("Running visual and metadata checks..."):
            if enable_ocr:
                from src.analysis.ocr import extract_text
                ocr_results.append(extract_text(primary_path))
                if second_path:
                    ocr_results.append(extract_text(second_path))
            report = screen_document(primary_path, ocr_result=ocr_results[0] if ocr_results else None)

        discrepancies = compare_identity_fields([
            result["full_text"] for result in ocr_results
        ]) if len(ocr_results) > 1 else []
        cross_document_discrepancies = compare_field_sets([
            report["fields"],
            *[
                extract_identity_fields(result["full_text"])
                for result in ocr_results[1:]
            ],
        ]) if len(ocr_results) > 1 else []
        discrepancies.extend(cross_document_discrepancies)
        if discrepancies:
            report["risk"] = calculate_risk(
                ela_score_value=report["ela_score"],
                copy_move_score=report["copy_move"]["score"],
                ocr_confidence=report["ocr"]["avg_confidence"] if report["ocr"] else 1.0,
                metadata_anomalies=len(report["metadata"]["anomalies"]),
                field_discrepancies=len(discrepancies),
                sdgi_anomalies=int(report["sdgi"]["detected"]),
            )
        report["field_discrepancies"] = discrepancies
        mrz_result = parse_mrz(ocr_results[0]["full_text"]) if ocr_results else {}
        report["mrz"] = mrz_result
        report["pipeline"] = build_inference_report(
            document_type=document_type,
            ocr=ocr_results[0] if ocr_results else None,
            mrz=mrz_result,
            tamper={"score": report["risk"]["components"]["visual_tamper"]},
            copy_move_score=float(report["copy_move"]["score"]),
            model_versions={"mrz": "icao9303-rules-v1", "tamper": "existing-forensics-v1"},
        )

        from src.rationale import build_rationale
        report["rationale"] = build_rationale(report)

        verdict_badge(report["risk"]["verdict"], caption=f"Confidence bucket: {report['risk']['confidence_bucket']}")
        st.info(report["rationale"])
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Overall risk", f"{report['risk']['score']:.1%}")
        metric2.metric("Copy-move score", f"{report['copy_move']['score']:.1%}")
        metric3.metric("ELA score", f"{report['ela_score']:.1%}")

        risk_components = report["risk"]["components"]
        risk_weights = report["risk"]["weights"]
        chart_data = {
            "Signal": list(risk_components.keys()),
            "Contribution": [
                risk_components[name] * risk_weights[name] for name in risk_components
            ],
        }
        fig = px.bar(
            chart_data, x="Contribution", y="Signal", orientation="h", range_x=[0, 0.3],
            title="Weighted contribution to overall risk",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=10, l=10, r=10), font=dict(family="Inter"),
        )
        fig.update_traces(marker_color="#4F8EF7")
        st.plotly_chart(fig, use_container_width=True)

        if video_file:
            st.warning("Video upload is accepted for future frame-quality analysis; it was not analyzed in this run.")

        with st.expander("MRZ and unified pipeline results"):
            if mrz_result.get("format") == "unknown":
                st.caption("No recognizable ICAO MRZ was found in the OCR output.")
            else:
                st.json(mrz_result)
            st.json(report["pipeline"])

        left, right = st.columns(2)
        with left:
            st.subheader("Evidence")
            with st.expander("Full evidence payload", expanded=False):
                st.json({
                    "metadata": report["metadata"],
                    "fields": report["fields"],
                    "field_discrepancies": report["field_discrepancies"],
                    "logic_checks": report["logic_checks"],
                    "read_trust": report["read_trust"],
                    "sdgi": report["sdgi"],
                    "evidence_regions": report["evidence_regions"],
                })
            if report["evidence_regions"]:
                from PIL import ImageDraw
                evidence_image = Image.open(primary_path).convert("RGB")
                drawer = ImageDraw.Draw(evidence_image)
                for region in report["evidence_regions"]:
                    drawer.rectangle(
                        (
                            region["x"],
                            region["y"],
                            region["x"] + region["width"],
                            region["y"] + region["height"],
                        ),
                        outline="red",
                        width=4,
                    )
                st.image(evidence_image, caption="Suspicious regions", use_container_width=True)
        with right:
            st.subheader("Audit trail")
            audit_path = Path(".verilogic/audit.jsonl")
            if st.button("🔗 Verify audit chain", key="verify_audit_chain"):
                from src.audit import verify_audit_chain
                verification = verify_audit_chain(audit_path)
                if verification["valid"]:
                    st.success(f"{verification['records']} records verified, chain intact")
                else:
                    st.error(f"BROKEN: {verification['error']}")
            audit_report = {
                "risk": report["risk"],
                "ela_score": report["ela_score"],
                "copy_move": report["copy_move"],
                "metadata": {
                    "format": report["metadata"]["format"],
                    "width": report["metadata"]["width"],
                    "height": report["metadata"]["height"],
                    "anomaly_count": len(report["metadata"]["anomalies"]),
                },
                "field_discrepancies": [item["field"] for item in discrepancies],
                "logic_checks": report["logic_checks"],
                "read_trust": report["read_trust"],
                "sdgi": report["sdgi"],
                "evidence_regions": report["evidence_regions"],
                "rationale": report["rationale"],
            }
            from src.audit import perceptual_hash
            from src.audit_similarity import find_similar_documents
            current_perceptual_hash = perceptual_hash(primary_path)
            json_path = export_json(report, Path(tempfile.gettempdir()) / "verilogic-report.json", document_path=primary_path)
            pdf_path = export_pdf(report, Path(tempfile.gettempdir()) / "verilogic-report.pdf", document_path=primary_path)
            dl1, dl2 = st.columns(2)
            dl1.download_button("⬇️ JSON report", json_path.read_bytes(), file_name="verilogic-report.json", mime="application/json", use_container_width=True)
            dl2.download_button("⬇️ PDF report", pdf_path.read_bytes(), file_name="verilogic-report.pdf", mime="application/pdf", use_container_width=True)
            audit_record = append_audit_record(
                audit_report,
                document_name=primary_file.name,
                document_path=primary_path,
                audit_path=audit_path,
            )
            similar = find_similar_documents(audit_path, current_perceptual_hash)
            if len(similar) > 1:
                st.warning("This document is visually similar to prior submissions; review for duplicate identity data.")
            st.success(f"Recorded decision {audit_record['record_hash'][:16]}…")
            st.caption("The full record is hash-chained in `.verilogic/audit.jsonl`.")
    else:
        empty_state("🛡️", "No screening run yet", "Upload a primary identity document above and run the screening")
