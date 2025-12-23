# =================================================================
# medical_pdf_agent/agent.py
# 
# =================================================================
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from typing import Dict, Any

# --- routing function---

def route_to_diabetes_model() -> Dict[str, str]:
    """
    Use this tool when the uploaded document is mainly about DIABETES.
    """
    return {
        "chosen_model": "diabetes_ml_analyst",
        "category": "diabetes",
        "message_fa": "این سند برای مدل تخصصی دیابت روت شد.",
    }


def route_to_cancer_model() -> Dict[str, str]:
    """
    Use this tool when the uploaded document is mainly about CANCER
    but NOT breast cancer specifically.
    """
    return {
        "chosen_model": "cancer_model",
        "category": "cancer",
        "message_fa": "این سند برای مدل سرطان عمومی روت شد.",
    }


def route_to_breast_cancer_model() -> Dict[str, str]:
    """
    Use this tool when the uploaded document is mainly about BREAST CANCER.
    """
    return {
        "chosen_model": "breast_cancer_model",
        "category": "breast_cancer",
        "message_fa": "این سند برای مدل سرطان سینه روت شد.",
    }


def ignore_document() -> Dict[str, Any]:
    """
    Use this tool when the uploaded document is NOT mainly about:
    - diabetes
    - cancer
    - breast cancer
    """
    return {
        "chosen_model": None,
        "category": "none",
        "message_fa": "سند مربوط به هیچ یک از دسته‌بندی‌های تخصصی نیست.",
    }


# --- main Agent rout---

root_agent = Agent(
    name="medical_pdf_router",
    model="gemini-2.5-flash", 
    description=(
        "Routes uploaded medical PDF documents to the correct specialist model: "
        "diabetes, general cancer, or breast cancer."
    ),
    instruction=(
        "You are a medical document ROUTING agent.\n\n"
        "The user will upload a medical PDF document (often in Persian or English).\n"
        "Your job is ONLY to decide which ONE of the three tools is best for this PDF:\n\n"
        "1) route_to_diabetes_model:\n"
        "   - Use when the MAIN topic is diabetes (type 1/2, blood sugar, insulin, HbA1c, etc.).\n\n"
        "2) route_to_cancer_model:\n"
        "   - Use when the MAIN topic is any cancer that is NOT specifically breast cancer,\n"
        "     or general oncology topics.\n\n"
        "3) route_to_breast_cancer_model:\n"
        "   - Use when the MAIN topic is breast cancer: diagnosis, treatment, imaging,\n"
        "     pathology, staging, surgery, chemo/hormone therapy focused on the breast.\n\n"
        "4) ignore_document:\n"
        "   - Use when the document is clearly NOT mainly about diabetes, cancer, "
        "     or breast cancer.\n"
        "   - In this case, the system wants to effectively return NO ANSWER to the user.\n\n"
        "VERY IMPORTANT ROUTING RULES:\n"
        "- Look at the CONTENT of the uploaded PDF (text, reports, imaging summaries, etc.).\n"
        "- Pick exactly ONE tool per request.\n"
        "- If it is clearly diabetes-focused -> use route_to_diabetes_model.\n"
        "- If it is clearly breast cancer-focused -> use route_to_breast_cancer_model.\n"
        "- If it is clearly another type of cancer or general oncology -> use route_to_cancer_model.\n"
        "- If it does NOT mainly match any of these, use ignore_document.\n"
        "- Always call EXACTLY ONE tool per request.\n"
        "- Do NOT answer medical questions yourself; always route via one tool.\n"
    ),
    tools=[
        FunctionTool(func=route_to_diabetes_model),
        FunctionTool(func=route_to_cancer_model),
        FunctionTool(func=route_to_breast_cancer_model),
        FunctionTool(func=ignore_document),
    ],
)
