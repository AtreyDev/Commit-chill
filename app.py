"""
DocAuth — AI Document Forgery Detection and Analysis
Streamlit multi-tab application.

Run with:
    streamlit run app.py

Tabs:
  1. Signature Verification    — Siamese network pair comparison
  2. Copy-Move Detection       — ORB+RANSAC / photoholmes
  3. Document Analysis         — ELA, edge detection, OCR, wavelet
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

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocAuth",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("DocAuth — AI Document Screening")
st.caption("Explainable OCR, MRZ validation, and image-forensics evidence for officer review")
st.caption("Powered by PyTorch Siamese networks · ORB+RANSAC · ELA · EasyOCR · PyWavelets")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dashboard, tab1, tab2, tab3, tab4 = st.tabs([
    "🌿  Optional wellness",
    "✍️  Signature Verification",
    "🔍  Copy-Move Detection",
    "📄  Document Analysis",
    "🛡️  AI Screening",
])


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


def _verdict_badge(verdict: str) -> None:
    colours = {"Authentic": "🟢", "Genuine": "🟢", "Suspicious": "🟡", "Forged": "🔴"}
    icon = colours.get(verdict, "⚪")
    st.markdown(f"## {icon} {verdict}")


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

    st.subheader("Wellness dashboard")
    st.markdown(
        "Track productivity while protecting work-life balance. This dashboard uses activity patterns, "
        "not medical diagnosis, to suggest healthier ways to work."
    )
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
    metric1.metric("Activity", metrics["total"])
    metric2.metric("Healthy streak", f"{healthy_streak(activities)} days")
    metric3.metric("Chill points", points)
    metric4.metric("Workload signal", f"{burnout['level']} · {burnout['score']:.0%}")

    chart_data = chart_rows(metrics)
    if chart_data:
        st.plotly_chart(px.bar(chart_data, x="date", y="activity", title="Activity by day"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Why this signal?")
        st.progress(burnout["score"], text=f"{burnout['level']} workload signal")
        for name, value in burnout["factors"].items():
            st.write(f"{name.replace('_', ' ').title()}: {value:.0%}")
        st.caption(burnout["disclaimer"])
    with right:
        st.subheader("Chill mode")
        if st.button("Take a 5-minute break", key="take_break"):
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
        notification_col.info(f"**{notification['title']}** — {notification['message']}")
        if action_col.button("Dismiss", key=f"dismiss_{notification['id']}"):
            st.session_state.dismissed_chill.add(notification["id"])
            st.rerun()

    st.subheader("Personal badges")
    if badges:
        st.dataframe(badges, hide_index=True, use_container_width=True)
    else:
        st.caption("Your first badge is waiting for a healthy-work habit.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — AI-Based Document Screening
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("AI-Based Fake Identity & Document Screening")
    st.markdown(
        "Combine visual tamper checks, copy-move detection, OCR confidence, "
        "metadata indicators, and cross-document identity checks into one explainable report."
    )

    document_type = st.selectbox(
        "Document type",
        ["Unknown", "Passport", "Identity card", "Driver license", "Other"],
        key="screen_document_type",
    )
    primary_file = st.file_uploader(
        "Primary identity document", type=["png", "jpg", "jpeg", "tiff", "bmp"], key="screen_primary"
    )
    second_file = st.file_uploader(
        "Second identity proof (optional)", type=["png", "jpg", "jpeg", "tiff", "bmp"], key="screen_second"
    )
    video_file = st.file_uploader(
        "Document video (optional; frame extraction not configured yet)",
        type=["mp4", "mov", "avi"],
        key="screen_video",
    )
    enable_ocr = st.checkbox("Run EasyOCR for identity fields", value=False, key="screen_ocr")
    st.info(
        "Face verification is not enabled until a consented, trained face-embedding checkpoint is configured."
    )

    if st.button("Run Document Screening", disabled=not primary_file):
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

        _verdict_badge(report["risk"]["verdict"])
        st.info(report["rationale"])
        st.caption(f"Confidence bucket: {report['risk']['confidence_bucket']}")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Overall risk", f"{report['risk']['score']:.1%}")
        metric2.metric("Copy-move score", f"{report['copy_move']['score']:.1%}")
        metric3.metric("ELA score", f"{report['ela_score']:.1%}")

        if video_file:
            st.warning("Video upload is accepted for future frame-quality analysis; it was not analyzed in this run.")
        with st.expander("MRZ and unified pipeline results"):
            if mrz_result.get("format") == "unknown":
                st.caption("No recognizable ICAO MRZ was found in the OCR output.")
            else:
                st.json(mrz_result)
            st.json(report["pipeline"])

        chart_data = {
            "Signal": list(report["risk"]["components"].keys()),
            "Contribution": [
                report["risk"]["components"][name] * report["risk"]["weights"][name]
                for name in report["risk"]["components"]
            ],
        }
        st.plotly_chart(
            px.bar(chart_data, x="Contribution", y="Signal", orientation="h", range_x=[0, 0.3]),
            use_container_width=True,
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Evidence")
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
            audit_path = Path(".docauth/audit.jsonl")
            if st.button("Verify audit chain", key="verify_audit_chain"):
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
            st.download_button("Download JSON report", json_path.read_bytes(), file_name="verilogic-report.json", mime="application/json")
            st.download_button("Download PDF report", pdf_path.read_bytes(), file_name="verilogic-report.pdf", mime="application/pdf")
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
            st.caption("The full record is hash-chained in `.docauth/audit.jsonl`. ")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Signature Verification
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Offline Signature Verification")
    st.markdown(
        "Upload a **reference** (enrolled) signature and a **query** (candidate) signature. "
        "The Siamese network compares their embeddings and determines if they match."
    )

    col_ref, col_qry = st.columns(2)
    with col_ref:
        ref_file = st.file_uploader(
            "Reference signature", type=["png", "jpg", "jpeg"], key="sig_ref"
        )
        if ref_file:
            st.image(ref_file, caption="Reference", use_container_width=True)

    with col_qry:
        qry_file = st.file_uploader(
            "Query signature", type=["png", "jpg", "jpeg"], key="sig_qry"
        )
        if qry_file:
            st.image(qry_file, caption="Query", use_container_width=True)

    weights_path = st.text_input(
        "Model weights path", value="weights/siamese_best.pt",
        help="Run `python -m src.signature.train` to generate weights."
    )

    if st.button("🔎 Verify Signatures", disabled=not (ref_file and qry_file)):
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

            _verdict_badge(result["verdict"])
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
    st.subheader("Copy-Move Forgery Detection")
    st.markdown(
        "Upload a document image. The detector identifies regions that have been "
        "copied and pasted within the same image using ORB keypoint matching and "
        "RANSAC geometric verification."
    )

    img_file = st.file_uploader(
        "Document image", type=["png", "jpg", "jpeg", "tiff"], key="cm_img"
    )

    if img_file:
        img_pil = Image.open(img_file).convert("RGB")
        st.image(img_pil, caption="Uploaded image", use_container_width=True)

        if st.button("🔎 Detect Copy-Move"):
            img_path = _save_upload(img_file)
            with st.spinner("Running copy-move detector..."):
                from src.copy_move.detector import detect_copy_move
                from src.copy_move.visualizer import overlay_heatmap, annotate_regions

                result = detect_copy_move(img_path)

            _verdict_badge(result["verdict"])
            m1, m2 = st.columns(2)
            m1.metric("Forgery Score", f"{result['score']:.1%}")
            m2.metric("Detection Method", result["method"])

            st.subheader("Detection Results")
            c1, c2 = st.columns(2)
            with c1:
                mask = result["mask"]
                if mask.any():
                    overlay = overlay_heatmap(np.array(img_pil), mask, alpha=0.4)
                    st.image(overlay, caption="Heatmap overlay", use_container_width=True)
                else:
                    st.info("No significant copy-move regions detected.")

            with c2:
                if result["heatmap"] is not None:
                    st.image(result["heatmap"], caption="Photoholmes heatmap", use_container_width=True)
                elif mask.any():
                    annotated = annotate_regions(np.array(img_pil), mask)
                    st.image(annotated, caption="Annotated regions", use_container_width=True)

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
    st.subheader("Document Analysis")
    st.markdown("Upload a document to run Error Level Analysis, edge detection, OCR, and wavelet decomposition.")

    doc_file = st.file_uploader(
        "Document image", type=["png", "jpg", "jpeg", "tiff", "bmp"], key="doc_img"
    )

    if doc_file:
        doc_pil = Image.open(doc_file).convert("RGB")
        st.image(doc_pil, caption="Uploaded document", use_container_width=True)

        analysis_options = st.multiselect(
            "Select analyses to run",
            ["Error Level Analysis (ELA)", "Edge Detection", "OCR", "Wavelet Decomposition"],
            default=["Error Level Analysis (ELA)", "Edge Detection"],
        )

        if st.button("▶ Run Analysis"):
            doc_path = _save_upload(doc_file)

            # ── ELA ───────────────────────────────────────────────────────────
            if "Error Level Analysis (ELA)" in analysis_options:
                st.subheader("Error Level Analysis")
                with st.spinner("Generating ELA map..."):
                    from src.analysis.ela import generate_ela, ela_score
                    ela_quality = st.session_state.get("ela_quality", 95)
                    ela_img = generate_ela(doc_pil, quality=ela_quality, scale=15)
                    score = ela_score(ela_img)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(doc_pil, caption="Original", use_container_width=True)
                with c2:
                    st.image(ela_img, caption="ELA Map", use_container_width=True)

                verdict = "Forged" if score > 0.08 else ("Suspicious" if score > 0.03 else "Authentic")
                _verdict_badge(verdict)
                st.metric("ELA Intensity Score", f"{score:.4f}")
                st.caption(
                    "Bright regions in the ELA map indicate areas that may have been "
                    "digitally manipulated. Uniform texture suggests an authentic image."
                )

            # ── Edge Detection ─────────────────────────────────────────────────
            if "Edge Detection" in analysis_options:
                st.subheader("Edge Detection")
                detector = st.selectbox(
                    "Detector", ["canny", "sobel", "laplacian", "prewitt_x", "prewitt_y"],
                    key="edge_det",
                )
                with st.spinner("Running edge detection..."):
                    from src.analysis.edge_detection import detect_all
                    edges = detect_all(doc_pil)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(doc_pil, caption="Original", use_container_width=True)
                with c2:
                    st.image(edges[detector], caption=f"{detector.capitalize()} edges", use_container_width=True)

            # ── OCR ───────────────────────────────────────────────────────────
            if "OCR" in analysis_options:
                st.subheader("Optical Character Recognition")
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
                        st.dataframe(df, use_container_width=True)

            # ── Wavelet ────────────────────────────────────────────────────────
            if "Wavelet Decomposition" in analysis_options:
                st.subheader("Wavelet Decomposition")
                col_w, col_l = st.columns(2)
                wavelet = col_w.selectbox("Wavelet", ["haar", "db1", "db4", "sym4"], key="wav_name")
                level = col_l.slider("Decomposition level", 1, 6, 3, key="wav_level")

                with st.spinner("Running wavelet decomposition..."):
                    from src.analysis.wavelet import decompose
                    wav_result = decompose(doc_pil, wavelet=wavelet, level=level)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(doc_pil, caption="Original", use_container_width=True)
                with c2:
                    st.image(wav_result["heatmap"], caption=f"{wavelet} detail heatmap (level {level})", use_container_width=True)

    st.divider()
    with st.expander("ℹ️  ELA quality setting"):
        quality = st.slider(
            "JPEG re-compression quality", min_value=70, max_value=99, value=95,
            help="Lower quality amplifies differences in manipulated regions.",
            key="ela_quality",
        )
