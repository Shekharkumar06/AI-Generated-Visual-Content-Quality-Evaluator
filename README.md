# AI-Generated Visual Content Quality Evaluator

A Python + Streamlit evaluation prototype for checking image-caption relevance, Hindi caption fluency, and text/OCR issues. It uses **pretrained models only**: no training, fine-tuning, or GPU is required.

## Features

- **Image-caption relevance:** pretrained multilingual CLIP (`sentence-transformers/clip-ViT-B-32-multilingual-v1`) with CPU inference.
- **Hindi fluency screening:** LanguageTool Hindi API when reachable; otherwise simple, explicitly-labelled heuristics.
- **Optional OCR:** uses `pytesseract` if the Tesseract system executable and Hindi/English language data are installed. Can compare extracted text with a user-supplied reference.
- **Human annotation categories:** mismatched caption, missing visual context, linguistic errors, culturally inappropriate content, textual/OCR inaccuracy, and other.
- Export evaluation records as CSV or JSON.

## Run on Windows

1. Install Python 3.10 or 3.11 (recommended).
2. Extract this ZIP and open a terminal in the project folder.
3. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Run on Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Pretrained model download

On the first evaluation with CLIP enabled, Sentence Transformers downloads the pretrained model from Hugging Face and caches it locally. Internet is needed for this first download. Subsequent runs can use the cache; to run fully offline on another machine, download/cache the model there first or copy the appropriate Hugging Face/Sentence Transformers cache. Model weights are not bundled in this ZIP because they are large and should be downloaded from their official model repository.

**No training happens.** `device="cpu"` is set explicitly. First-time inference can be slow and CPU memory requirements vary by machine.

## OCR setup (optional)

OCR is disabled by default. Install Tesseract OCR separately and ensure Hindi (`hin`) and English (`eng`) language packs are available, then install the `pytesseract` Python package from `requirements.txt`. If Tesseract is missing, the app will explain that OCR could not run.

## Important limitations

- CLIP cosine similarity is a raw similarity signal, **not a probability**. Threshold labels in the UI are heuristic and must be calibrated on human-labelled examples for research or production.
- The Hindi fallback checks are not a substitute for a full Hindi grammar model. LanguageTool availability and the quality/coverage of its Hindi rules can vary.
- Cultural appropriateness and nuanced Hindi fluency require human review. The app supports annotation; it does not automatically make reliable cultural-safety decisions.
- OCR exact-match is strict after whitespace/case normalization; it does not compute CER/WER.
- Uploaded content is processed locally by Streamlit except for optional calls to LanguageTool's public API and the first-time pretrained model download. Avoid submitting confidential images/captions when using external services.

## Project layout

```text
AI_Generated_Visual_Content_Quality_Evaluator/
├── app.py
├── requirements.txt
├── README.md
└── models/   # reserved for optional local model/cache notes; weights are downloaded by Sentence Transformers
```
