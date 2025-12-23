# =================================================================
# =================================================================

import os
import sys
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException

# 
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# 
from medical_pdf_agent.agent import root_agent as router_agent
from my_agent.agent import diabetes_analyst_agent

# 
from google.adk.runners import InMemoryRunner, RunConfig
from google.genai.types import Content, Part, Blob

APP_NAME = "medical_pdf_router_app"

# 
router_runner = InMemoryRunner(agent=router_agent, app_name=APP_NAME)
diabetes_runner = InMemoryRunner(agent=diabetes_analyst_agent, app_name=APP_NAME)

# 
def run_agent_on_pdf(runner, instruction, pdf_path, session_id):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    content = Content(
        role="user",
        parts=[
            Part(text=instruction),
            Part(inline_data=Blob(mime_type="application/pdf", data=pdf_bytes)),
        ],
    )

    events = runner.run(
        user_id="user_pdf",
        session_id=session_id,
        new_message=content,
        run_config=RunConfig(save_input_blobs_as_artifacts=True),
    )

    full_text = ""
    for ev in events:
        if ev.content and ev.content.parts:
            for p in ev.content.parts:
                if getattr(p, "text", None):
                    full_text += p.text
    return full_text

# 
def route_pdf(pdf_path):
    session_id = str(uuid.uuid4())

    #
    router_instruction = (
        "این سند پزشکی را بررسی کن و فقط یکی از ابزارهای روتینگ زیر را فراخوانی کن:\n"
        "- route_to_diabetes_model\n"
        "- route_to_cancer_model\n"
        "- route_to_breast_cancer_model\n"
        "- ignore_document\n"
        "هیچ متن اضافه‌ای نده؛ فقط ابزار مناسب را صدا بزن."
    )

    router_output = run_agent_on_pdf(router_runner, router_instruction, pdf_path, session_id)
    tool_name = None
    chosen_model = None

    # 
    if "route_to_diabetes_model" in router_output or '"chosen_model": "diabetes_ml_analyst"' in router_output:
        tool_name = "route_to_diabetes_model"
        chosen_model = "diabetes_ml_analyst"
    elif "route_to_cancer_model" in router_output:
        tool_name = "route_to_cancer_model"
        chosen_model = "cancer_model"
    elif "route_to_breast_cancer_model" in router_output:
        tool_name = "route_to_breast_cancer_model"
        chosen_model = "breast_cancer_model"

    # 
    agent_output = None
    if chosen_model == "diabetes_ml_analyst":
        agent_instruction = "تمام 148 ویژگی را از PDF استخراج کن و پیش‌بینی کن. فقط خروجی نهایی ابزار را برگردان."
        agent_output = run_agent_on_pdf(diabetes_runner, agent_instruction, pdf_path, session_id)
    elif chosen_model:
        agent_output = f"سند مربوط به {chosen_model} است."
    else:
        agent_output = "سند مربوط به دیابت یا سرطان نیست."

    return {
        "session_id": session_id,
        "router": {
            "tool_name": tool_name,
            "chosen_model": chosen_model,
            "full_output": router_output,
        },
        "agent_output": agent_output,
    }

# ---- FastAPI ----
app = FastAPI(title="Medical PDF Router")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(400, "فقط PDF مجاز است.")

    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        return route_pdf(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
