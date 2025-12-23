
# 🩺 Medical PDF Router  
### Intelligent Medical Document Routing with Google Gemini ADK

An intelligent medical document routing and analysis system built using **Google Gemini Agent Development Kit (ADK)**.  
This project classifies medical PDF documents and routes diabetes-related files to a specialized **Machine Learning (ML)** model for accurate clinical prediction.

---

## 🚀 Overview

The system automatically:
- Analyzes medical PDF documents
- Classifies them into predefined medical categories
- Routes diabetes-related documents to a dedicated ML-powered agent
- Extracts structured medical features and generates predictions

This architecture demonstrates **multi-agent orchestration**, **LLM-powered document understanding**, and **ML model integration** in a real-world medical use case.

---

## 🧠 System Architecture

The workflow consists of three main components:

### 1️⃣ Primary Router Agent (`medical_pdf_agent`)
- Analyzes the input PDF
- Classifies it into one of the following categories:
  - Diabetes
  - General Cancer
  - Breast Cancer
  - Irrelevant / Non-medical

---

### 2️⃣ Orchestrator (`main.py`)
- Acts as the system entry point
- Receives the classification result from the router agent
- If **Diabetes** is detected, forwards the same PDF to the specialized diabetes agent

---

### 3️⃣ Diabetes Specialist Agent (`my_agent`)
- Uses **Gemini** to extract **148 structured medical features** from the PDF
- Loads a fine-tuned ML model (`final_diabetes_model.pkl`)
- Returns the final clinical prediction

---

## 📁 Project Structure

```

MEDICAL_PDF_ROUTER/
├── .venv/                         # Virtual environment
├── main.py                        # Orchestrator and entry point
├── requirements.txt               # Python dependencies
├── diabetes_sample.pdf            # Sample medical PDF for testing
├── medical_pdf_agent/             # Router agent package
│   └── agent.py
└── my_agent/                      # Diabetes specialist agent package
├── agent.py                   # Feature extraction + ML inference logic
└── final_diabetes_model.pkl   # Fine-tuned ML model

````

---

## ⚙️ Setup & Installation

### 1️⃣ Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
````

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Configure Gemini API Key

Set your **Gemini API Key** as an environment variable:

```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

(Windows PowerShell)

```powershell
setx GEMINI_API_KEY "YOUR_API_KEY_HERE"


---

### 4️⃣ ML Model Configuration

The ML model path is already configured **relative to the project structure** inside:

```
my_agent/agent.py
```

Make sure the following file exists:

```
my_agent/final_diabetes_model.pkl
```

---

## ▶️ Running the Application

1. Place a test medical PDF in the project root
   (example: `diabetes_sample.pdf`)
2. Run the orchestrator:

```bash
python main.py
```

---

## ✅ Expected Output

A successful run will look similar to:

```
🔬 Step 1: Analyzing medical PDF...
📄 Category detected: Diabetes

🔬 Step 2: Routing to specialized diabetes agent...

✨ Diabetes Specialist Result:
Predicted Fasting Plasma Glucose (mg/dl): [numeric value]


## 🧪 Technologies Used

* **Google Gemini ADK**
* **Python**
* **Large Language Models (LLMs)**
* **Machine Learning (Scikit-learn / Pickle model)**
* **PDF Document Processing**
* **Multi-Agent Architecture**

---

## 📌 Use Cases

* Intelligent medical document triage
* Clinical decision support systems
* AI-powered medical data extraction
* Demonstration of agent-based LLM orchestration

---

## 📄 License

This project is provided for educational and research purposes.

---

✨ *Built with Gemini ADK and a multi-agent AI architecture*



