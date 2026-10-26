<div align="center">

# 🎓 ScholarMind AI

### AI Research Assistant

Turn a research paper into a short, organised report with **key concepts, findings, takeaways and citations**, then ask follow-up questions about the paper.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.3%2070B-F55036)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

</div>

---

## 📖 Overview

**ScholarMind AI** is a web application built with Python and Streamlit. The user pastes the text of a research paper or uploads a file, and the app sends it to an AI model. The model returns a structured analysis of the paper. The user can then ask follow-up questions and download the report in several formats.

> **Internship task:** Develop an AI-powered research assistant that summarizes research articles or study materials into concise reports, identifying important concepts, generating key takeaways and presenting information in an organized format.
>
> **Level:** Intermediate Level – Project 3 · **Domain:** Artificial Intelligence · **Organisation:** CyberSNova

---

## ✨ Features

| Feature | Description |
|---|---|
| 📝 **Flexible input** | Paste paper text or upload `.pdf`, `.txt` or `.md` files |
| 🔬 **3 research modes** | Skim, Standard and Deep Analysis to control depth |
| 🧩 **Structured report** | Plain-language summary, key concepts with definitions, research question, methodology, key findings, contributions, limitations, future work, keywords, takeaways and a short critical review |
| 💬 **Paper Q&A** | Ask follow-up questions about the analysed paper |
| 📚 **Citations** | Generates APA, MLA and BibTeX citations |
| 🃏 **Flashcards** | Study flashcards exported as CSV |
| 🌐 **Bilingual output** | Output language: English or Roman Urdu |
| 🗂️ **Paper history** | Keeps up to 20 analysed papers in the sidebar |
| 🔁 **Model fallback** | Order of models chosen to stay within Groq rate limits |
| ✂️ **Smart truncation** | Very long papers are shortened intelligently to fit the model limit |
| 📤 **Multi-format export** | Markdown, PDF, JSON, flashcards CSV and citation files |

### 🔬 Research Modes

| Mode | What you get |
|---|---|
| ⚡ **Skim** | Quick one-minute overview: title, main claim and verdict |
| 📄 **Standard** | Balanced analysis of all sections at moderate depth |
| 🔬 **Deep Analysis** | Exhaustive breakdown: methodology, metrics, limitations and future work |

---

## 🛠️ Tech Stack

| Tool | Used for |
|---|---|
| **Python 3** | Main programming language |
| **Streamlit** | Web interface: tabs, text box, file upload, chat box, sidebar settings |
| **Groq API (Llama 3.3 70B)** | AI model that analyses the paper and answers questions |
| **OpenAI Python library** | Client used to call Groq's OpenAI-compatible API |
| **PyMuPDF** | Reading text from PDF files |
| **pandas** | Building the flashcards CSV |
| **fpdf2** | Generating the PDF report |

---

## 📁 Project Structure

```text
ScholarMind-AI/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .streamlit/
    └── secrets.toml     # (optional, not committed) API key
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- A free [Groq API key](https://console.groq.com/keys)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ScholarMind-AI.git
cd ScholarMind-AI
```

### 2. Create a virtual environment *(recommended)*
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt`:
```text
streamlit
openai
pandas
pymupdf
fpdf2
```

> PDF reading and PDF export are optional features. If a library is missing, the app still runs and disables that feature.

### 4. Set your Groq API key

**Option A – Environment variable**
```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your_api_key_here"

# macOS / Linux
export GROQ_API_KEY="your_api_key_here"
```

**Option B – Streamlit secrets** (`.streamlit/secrets.toml`)
```toml
GROQ_API_KEY = "your_api_key_here"
```

### 5. Run the app
```bash
streamlit run app.py
```
The app opens at **http://localhost:8501**.

---

## 🖥️ Usage

1. Open the **📝 Input** tab and choose **Paste text** or **Upload PDF**.
2. Pick a **research mode** and the output language in the sidebar.
3. Run the analysis and open the **📋 Report** tab.
4. Ask follow-up questions in the **💬 Q&A** tab.
5. Download results from the **📤 Export** tab.

### 📤 Export Formats

| Format | File |
|---|---|
| Markdown report | `research_report.md` |
| PDF report | `research_report.pdf` |
| JSON data | `research_data.json` |
| Flashcards | `flashcards.csv` |
| APA citation | `citation_apa.txt` |
| BibTeX citation | `citation.bib` |

### ⚙️ Model Selection

Groq limits how many tokens you can send per minute (TPM). The sidebar model list is ordered with this in mind:

| Model | Note |
|---|---|
| `llama-3.1-8b-instant` | Highest limit, safest for large papers |
| `llama-3.3-70b-versatile` | Best quality |
| Others | Lower limits |

**Limits:** about 20,000 characters sent comfortably, up to 45,000 characters maximum · 20 papers kept in history · 12 Q&A turns of context.

---


---

## ⚠️ Limitations

- Very long papers are shortened before analysis, so some detail may be left out.
- Scanned image-only PDFs need OCR first.
- Citations are generated from the paper text and should be checked before use.
- The PDF export uses a basic Latin character set, so some special characters may be simplified.
- Free Groq usage is subject to rate limits.

---

## 🔮 Future Improvements

- OCR for scanned papers
- Comparing multiple papers side by side
- Direct import from arXiv or DOI links

---

## 👤 Author

**Muhammad Numan**
AI Engineering Intern at **[CyberSNova](https://cybersnova.co.in)**


---

<div align="center">

⭐ If you found this project useful, consider giving it a star.

</div>
