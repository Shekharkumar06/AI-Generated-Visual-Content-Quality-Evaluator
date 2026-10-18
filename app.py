import io, re, json, time
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="AI-Generated Visual Content Quality Evaluator", page_icon="🖼️", layout="wide")

MODEL_ID = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

@st.cache_resource(show_spinner=False)
def load_clip_model():
    """Load a pretrained multilingual CLIP model; no fine-tuning/training is performed."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID, device="cpu")

def clip_relevance(image: Image.Image, caption: str):
    model = load_clip_model()
    image_vec = model.encode([image.convert("RGB")], convert_to_tensor=True, normalize_embeddings=True)
    text_vec = model.encode([caption], convert_to_tensor=True, normalize_embeddings=True)
    # Cosine similarity is the useful raw score; don't present it as a calibrated probability.
    import torch
    score = float(torch.sum(image_vec * text_vec, dim=1).item())
    return score

def language_tool_check(text: str):
    """Optional LanguageTool Public API check. Gracefully falls back when unavailable."""
    try:
        import requests
        response = requests.post(
            "https://api.languagetool.org/v2/check",
            data={"text": text, "language": "hi"}, timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        matches = []
        for item in payload.get("matches", []):
            ctx = item.get("context", {})
            matches.append({
                "message": item.get("message", "Possible issue"),
                "suggestions": ", ".join(x.get("value", "") for x in item.get("replacements", [])[:4]),
                "context": ctx.get("text", ""),
                "offset": item.get("offset", 0),
                "length": item.get("length", 0),
                "rule": item.get("rule", {}).get("id", ""),
            })
        return matches, "LanguageTool Hindi check"
    except Exception:
        # Lightweight heuristics, explicitly labelled as heuristics, not a Hindi grammar model.
        matches = []
        if re.search(r"\s{2,}", text):
            matches.append({"message": "Multiple consecutive spaces", "suggestions": "Use a single space", "context": text, "offset": 0, "length": 0, "rule": "HEURISTIC_SPACES"})
        if re.search(r"[A-Za-z]{5,}", text):
            matches.append({"message": "Possible Latin-script word in Hindi caption; check code-mixing/spelling", "suggestions": "Review manually", "context": text, "offset": 0, "length": 0, "rule": "HEURISTIC_CODEMIX"})
        if len(text.strip()) < 8:
            matches.append({"message": "Caption is very short; fluency/context may be hard to judge", "suggestions": "Add a complete descriptive phrase", "context": text, "offset": 0, "length": 0, "rule": "HEURISTIC_SHORT"})
        return matches, "Offline heuristics fallback (not a full Hindi grammar checker)"

def ocr_text(image: Image.Image):
    try:
        import pytesseract
        text = pytesseract.image_to_string(image, lang="hin+eng")
        return text.strip(), None
    except Exception as exc:
        return "", str(exc)

def simple_category_flags(caption, issues, relevance):
    flags = []
    if relevance < 0.20:
        flags.append("Possible mismatched caption")
    if len(caption.split()) < 5:
        flags.append("Possible missing visual context")
    if issues:
        flags.append("Possible linguistic errors")
    return flags or ["No automatic flag; manual review still recommended"]

st.title("AI-Generated Visual Content Quality Evaluator")
st.caption("CPU-first evaluation prototype • pretrained models only • no model training or fine-tuning")
st.info("Image-caption relevance uses multilingual CLIP embeddings. Hindi fluency checks use LanguageTool when reachable, otherwise transparent heuristics. Scores are screening signals—not ground truth or calibrated probabilities.")

with st.sidebar:
    st.header("Evaluation settings")
    run_clip = st.checkbox("Run pretrained CLIP relevance", value=True)
    run_language = st.checkbox("Check Hindi caption fluency", value=True)
    run_ocr = st.checkbox("Try OCR for visible text", value=False)
    st.caption("First CLIP run downloads pretrained weights from Hugging Face. After caching, inference runs on CPU.")

left, right = st.columns([1, 1])
with left:
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=uploaded.name, use_container_width=True)
with right:
    caption = st.text_area("Hindi caption to evaluate", placeholder="उदाहरण: एक बच्चा पार्क में गेंद से खेल रहा है।", height=130)
    expected_text = st.text_area("Expected visible text (optional; for manual OCR comparison)", placeholder="If the image contains text, enter the reference text here.", height=90)
    note = st.text_input("Reviewer note (optional)")
    categories = st.multiselect("Human annotation categories", [
        "Mismatched caption", "Missing visual context", "Linguistic errors", "Culturally inappropriate content", "Textual/OCR inaccuracy", "Other"
    ])
    reviewed_by = st.text_input("Reviewer ID / initials (optional)")

if st.button("Evaluate content", type="primary", use_container_width=True):
    if uploaded is None:
        st.error("Please upload an image first.")
    elif not caption.strip():
        st.error("Please enter a caption first.")
    else:
        results = {"timestamp": datetime.now().isoformat(timespec="seconds"), "image_name": uploaded.name, "caption": caption.strip()}
        c1, c2, c3 = st.columns(3)
        if run_clip:
            with st.spinner("Loading pretrained multilingual CLIP on CPU and comparing image/caption…"):
                try:
                    score = clip_relevance(image, caption.strip())
                    results["clip_cosine_similarity"] = round(score, 4)
                    c1.metric("Image-caption similarity", f"{score:.3f}")
                    if score >= 0.30:
                        c1.caption("Higher relative alignment signal (threshold is heuristic)")
                    elif score >= 0.20:
                        c1.caption("Borderline alignment signal; inspect manually")
                    else:
                        c1.caption("Low alignment signal; possible mismatch")
                except Exception as exc:
                    st.error(f"CLIP could not run: {exc}")
                    st.caption("Check internet access for first-time model download and install requirements. No training is needed.")
        issues, checker_name = ([], "Not run")
        if run_language:
            with st.spinner("Checking Hindi caption fluency…"):
                issues, checker_name = language_tool_check(caption.strip())
            results["fluency_checker"] = checker_name
            results["fluency_issue_count"] = len(issues)
            c2.metric("Potential language issues", len(issues))
            c2.caption(checker_name)
        ocr = ""
        if run_ocr:
            with st.spinner("Attempting OCR on image…"):
                ocr, ocr_error = ocr_text(image)
            if ocr_error:
                st.warning("OCR unavailable. Install Tesseract OCR with Hindi language data and the pytesseract Python package.")
                st.caption(f"Technical detail: {ocr_error}")
            else:
                st.subheader("OCR text extracted from image")
                st.text_area("Extracted text", value=ocr or "No text detected", height=100, disabled=True)
                if expected_text.strip():
                    norm = lambda x: re.sub(r"\s+", " ", x.strip().lower())
                    exact = norm(ocr) == norm(expected_text)
                    results["ocr_reference_match"] = exact
                    c3.metric("OCR reference match", "Match" if exact else "Review")
                    results["ocr_extracted_text"] = ocr
                else:
                    c3.metric("OCR text detected", "Yes" if ocr else "No")
                    results["ocr_extracted_text"] = ocr
        if not run_ocr:
            c3.metric("OCR check", "Not run")
        score_for_flags = results.get("clip_cosine_similarity", 0.5)
        auto_flags = simple_category_flags(caption.strip(), issues, score_for_flags)
        st.subheader("Automated screening")
        for flag in auto_flags:
            st.write(f"- {flag}")
        if issues:
            st.subheader("Potential language issues")
            st.dataframe(pd.DataFrame(issues), use_container_width=True, hide_index=True)
        results["auto_flags"] = "; ".join(auto_flags)
        results["human_annotation_categories"] = "; ".join(categories)
        results["reviewer_note"] = note
        results["reviewer_id"] = reviewed_by
        st.session_state["last_result"] = results
        st.success("Evaluation completed. Review automated signals and confirm the final labels manually.")

if "last_result" in st.session_state:
    st.subheader("Export evaluation record")
    record = st.session_state["last_result"]
    df = pd.DataFrame([record])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download result as CSV", data=df.to_csv(index=False).encode("utf-8-sig"), file_name="visual_quality_evaluation.csv", mime="text/csv")
    st.download_button("Download result as JSON", data=json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"), file_name="visual_quality_evaluation.json", mime="application/json")

with st.expander("Annotation guide"):
    st.markdown("""
    - **Mismatched caption:** caption describes objects/actions not supported by the image.
    - **Missing visual context:** important subject, action, setting, relation, or visible text is omitted.
    - **Linguistic errors:** grammar, spelling, agreement, punctuation, unnatural phrasing, or unclear Hindi.
    - **Culturally inappropriate content:** stereotype, insensitive wording, or a culturally misleading interpretation; always use human review.
    - **Textual/OCR inaccuracy:** visible text is omitted, misread, or represented incorrectly.

    CLIP similarity is a raw cosine similarity and should be calibrated against a representative, human-labelled validation set before setting production thresholds. This app intentionally does not train or fine-tune any model.
    """)
