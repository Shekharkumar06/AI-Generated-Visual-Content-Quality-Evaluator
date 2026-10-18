import io
import os
import re

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Visual Evaluator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_ID = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


@st.cache_resource(show_spinner=False)
def load_relevance_model():
    """
    Load the model once per running app process.

    Prefer cached local weights first. If unavailable, try one online download.
    Return the error instead of raising it so a failed download is cached too;
    this prevents retrying a timed-out download on every Streamlit rerun.
    """
    try:
        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(
                MODEL_ID,
                device="cpu",
                local_files_only=True,
            )
            return model, None
        except Exception:
            # Only attempt a network download once if no complete local cache exists.
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
            os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
            model = SentenceTransformer(MODEL_ID, device="cpu")
            return model, None
    except Exception:
        return None, "The image comparison model is not available locally."


def offline_relevance_estimate(image: Image.Image, caption: str) -> int:
    """
    Provide a clearly labelled, coarse offline estimate when model weights
    cannot be loaded. This is not semantic image-caption matching.
    """
    width, height = image.size
    word_count = len(caption.split())
    has_hindi = bool(re.search(r"[\\u0900-\\u097F]", caption))
    score = 35
    if word_count >= 4:
        score += 20
    elif word_count >= 2:
        score += 10
    if has_hindi:
        score += 15
    if width >= 256 and height >= 256:
        score += 20
    elif width >= 100 and height >= 100:
        score += 10
    if caption.strip().endswith(("।", ".", "!", "?")):
        score += 5
    return max(0, min(95, score))


def relevance_score(image: Image.Image, caption: str) -> tuple[float | None, int | None, str]:
    """
    Return (raw cosine score, offline estimate, mode).
    A real semantic score is returned only when the pretrained model is available.
    """
    model, _load_error = load_relevance_model()
    if model is None:
        return None, offline_relevance_estimate(image, caption), "offline"

    try:
        import torch

        image_vector = model.encode(
            [image.convert("RGB")],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        text_vector = model.encode(
            [caption],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        raw_score = float(torch.sum(image_vector * text_vector, dim=1).item())
        return raw_score, None, "model"
    except Exception:
        return None, offline_relevance_estimate(image, caption), "offline"


def check_hindi_fluency(text: str) -> tuple[int, str, list[str]]:
    """
    Do a free, best-effort language check.

    This deliberately avoids a mandatory external API. The lightweight checks
    flag obvious formatting issues and are not presented as a full grammar model.
    """
    issues: list[str] = []

    if re.search(r"\s{2,}", text):
        issues.append("Extra spaces found. Use a single space between words.")
    if re.search(r"[ \t]+[।,!?]", text):
        issues.append("Remove spaces before punctuation.")
    if len(text.strip()) < 8:
        issues.append("The caption is very short; fluency may be difficult to assess.")
    if not re.search(r"[\u0900-\u097F]", text):
        issues.append("No Devanagari characters detected. Check Hindi spelling or script.")

    # A transparent consistency score based on detected formatting flags.
    # It is a heuristic score, not a linguistic probability.
    score = max(0, round(100 * (1 - min(len(issues), 4) / 5)))
    summary = "Looks good" if not issues else f"{len(issues)} suggestion(s) to review"
    return score, summary, issues


def extract_visible_text(image: Image.Image) -> tuple[str, str | None]:
    """Try optional local OCR; report a friendly message if unavailable."""
    try:
        import pytesseract

        return pytesseract.image_to_string(image, lang="hin+eng").strip(), None
    except Exception:
        return "", "Text recognition is unavailable in this deployment."


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def get_overall_score(relevance_score_0_100: int, fluency: int | None) -> int:
    """Combine the displayed relevance score with fluency when enabled."""
    rel = max(0, min(100, relevance_score_0_100))
    if fluency is None:
        return round(rel)
    return round((rel + fluency) / 2)


# Theme-aware polish: rely on Streamlit's theme variables for all surfaces and text.
st.markdown(
    """
    <style>
      .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
      }
      .ve-eyebrow {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--primary-color);
        margin-bottom: 0.45rem;
      }
      .ve-title {
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 750;
        letter-spacing: -0.045em;
        line-height: 1.08;
        margin: 0;
      }
      .ve-subtitle {
        margin-top: 0.65rem;
        font-size: 1.05rem;
        color: var(--text-color);
        opacity: 0.72;
      }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
        box-shadow: 0 5px 24px rgba(55, 65, 81, 0.07);
      }
      div[data-testid="stFileUploader"] section {
        border-radius: 12px;
      }
      div.stButton > button[kind="primary"] {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.9rem;
      }
      div[data-testid="stMetric"] {
        border: 1px solid var(--secondary-background-color);
        border-radius: 14px;
        padding: 1rem 1.1rem;
      }
      .ve-footnote {
        font-size: 0.84rem;
        opacity: 0.68;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="ve-eyebrow">Image &amp; caption analysis</div>
    <h1 class="ve-title">Visual Evaluator</h1>
    <div class="ve-subtitle">AI Powered Image &amp; Text Quality Check</div>
    """,
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1, 1], gap="large")

with left:
    with st.container(border=True):
        st.subheader("1. Upload image")
        uploaded = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            help="Supported formats: PNG, JPG, JPEG, WebP and BMP.",
            key="visual_evaluator_upload",
        )
        image = None
        if uploaded is not None:
            try:
                image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
                st.image(image, caption=uploaded.name, use_container_width=True)
                st.caption(f"{image.width} × {image.height} px")
            except Exception:
                st.error("This image could not be opened. Please try another file.")
                image = None
        else:
            st.markdown("Upload an image to see its preview here.")
            st.caption("A clear image helps produce a more useful comparison.")

with right:
    with st.container(border=True):
        st.subheader("2. Add caption")
        caption = st.text_area(
            "Hindi caption",
            placeholder="उदाहरण: एक बच्चा पार्क में गेंद से खेल रहा है।",
            height=150,
            help="Enter the Hindi caption that describes the uploaded image.",
            key="visual_evaluator_caption",
        )
        check_grammar = st.checkbox(
            "Check Hindi Grammar & Fluency",
            value=True,
            help="Runs a free, lightweight Hindi text-quality check.",
        )
        with st.expander("Advanced - OCR Comparison (Optional)"):
            expected_text = st.text_area(
                "Expected visible text",
                placeholder="If the image contains text, enter the correct text here.",
                height=100,
                help="Optional reference text to compare with text found in the image.",
                key="visual_evaluator_expected_text",
            )
            st.caption("OCR comparison is attempted only when reference text is provided.")

        evaluate = st.button("Evaluate", type="primary", use_container_width=True)

if evaluate:
    if image is None:
        st.error("Please upload a valid image first.")
    elif not caption.strip():
        st.error("Please enter a Hindi caption first.")
    else:
        current_caption = caption.strip()
        fluency = None
        fluency_summary = "Skipped"
        fluency_issues: list[str] = []
        ocr_match = None
        evaluation_error = None

        with st.spinner("Evaluating image and caption…"):
            relevance_raw, offline_score, evaluation_mode = relevance_score(
                image, current_caption
            )

        if check_grammar:
            fluency, fluency_summary, fluency_issues = check_hindi_fluency(
                current_caption
            )

        if evaluation_mode == "model" and relevance_raw is not None:
            # Map raw cosine similarity to a readable 0–100 alignment indicator.
            # This is an alignment index, not a calibrated probability.
            relevance_display = round(
                max(0.0, min(1.0, (relevance_raw + 1.0) / 2.0)) * 100
            )
        else:
            relevance_display = int(offline_score or 0)

        overall = get_overall_score(relevance_display, fluency)

        st.write("")
        st.subheader("Evaluation results")
        metric_columns = st.columns(3, gap="medium")
        with metric_columns[0]:
            st.metric("Relevance Score", f"{relevance_display}/100")
        with metric_columns[1]:
            st.metric("Fluency Score", f"{fluency}/100" if fluency is not None else "Skipped")
        with metric_columns[2]:
            st.metric("Overall", f"{overall}/100")

        if evaluation_mode == "offline":
            st.warning(
                "Offline estimate shown. Semantic image-caption matching needs the "
                "comparison model weights; this estimate uses only local image and "
                "caption properties and may not reflect actual relevance."
            )
        elif relevance_display >= 70:
            st.success("The caption appears well aligned with the image.")
        elif relevance_display >= 50:
            st.info("The caption may be partially aligned. Review the details.")
        else:
            st.warning("The caption may not closely match the image. Please review it.")

        if check_grammar:
            with st.container(border=True):
                st.markdown("**Hindi grammar & fluency**")
                st.write(fluency_summary)
                if fluency_issues:
                    for issue in fluency_issues:
                        st.markdown(f"- {issue}")
                st.caption(
                    "This free check looks for simple text issues and is not a "
                    "complete assessment of Hindi grammar or meaning."
                )

        if expected_text.strip():
            with st.container(border=True):
                st.markdown("**OCR comparison**")
                with st.spinner("Checking text visible in the image…"):
                    ocr_text, ocr_error = extract_visible_text(image)
                if ocr_error:
                    st.info(
                        "Text recognition isn't available in this deployment. "
                        "Your reference text was not evaluated."
                    )
                else:
                    ocr_match = normalize_text(ocr_text) == normalize_text(expected_text)
                    st.metric(
                        "Visible text comparison",
                        "Match" if ocr_match else "Needs review",
                    )
                    if ocr_text:
                        st.text_area(
                            "Text found in image",
                            value=ocr_text,
                            height=100,
                            disabled=True,
                        )
                    else:
                        st.caption("No text was detected in the image.")

        st.session_state["visual_evaluator_result"] = {
            "relevance_score": relevance_display,
            "fluency_score": fluency,
            "overall_score": overall,
            "fluency_summary": fluency_summary,
            "ocr_comparison": (
                ("Match" if ocr_match else "Needs review")
                if ocr_match is not None
                else "Unavailable or not requested"
            ),
        }

st.markdown("---")
st.markdown(
    '<div class="ve-footnote">Scores are guidance for review, not guarantees of correctness. '
    'Image-caption relevance is an alignment indicator. If model weights are unavailable, '
    'the app uses a clearly labelled basic offline estimate; fluency checks are rule-based.</div>',
    unsafe_allow_html=True,
)
