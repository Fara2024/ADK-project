# =================================================================
# =================================================================
import os
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

# 
try:
    # 
    from medical_pdf_agent.agent import root_agent as initial_router_agent 
    print("✅ روتر اولیه با موفقیت ایمپورت شد.")
except ImportError as e:
    print(f"❌ خطا در ایمپورت روتر اولیه: {e}. مطمئن شوید 'medical_pdf_agent/agent.py' موجود است.")
    initial_router_agent = None


#
try:
    # 
    from my_agent.agent import diabetes_analyst_agent 
    print("✅ Agent تخصصی دیابت با موفقیت ایمپورت شد.")
except ImportError as e:
    print(f"❌ خطا در ایمپورت Agent دیابت: {e}. مطمئن شوید 'my_agent/agent.py' و مدل ML موجود هستند.")
    diabetes_analyst_agent = None


# 

def process_medical_pdf(pdf_file_path: str) -> str:
    """
    مدیریت کل جریان کار: دسته‌بندی با روتر اولیه و در صورت نیاز، 
    استخراج داده و پیش‌بینی تخصصی با Agent دیابت.
    """
    if initial_router_agent is None or diabetes_analyst_agent is None:
        return "\n❌ اجرای Agent به دلیل خطای ایمپورت/بارگذاری قبلی امکان‌پذیر نیست."

    if not os.path.exists(pdf_file_path):
        return f"خطا: فایل در مسیر {pdf_file_path} یافت نشد."

    print(f"\n🚀 شروع تحلیل برای فایل: {pdf_file_path}")
    
    # 1.
    try:
        print("🔍 مرحله 1: ارسال به روتر اولیه برای دسته‌بندی...")
        
        # 
        router_result = initial_router_agent.run(
            prompt="لطفاً این سند را تحلیل کنید و مدل تخصصی مناسب را انتخاب کنید. پاسخ شما فقط باید شامل فراخوانی ابزار باشد.",
            files=[pdf_file_path]
        )
        
        # 2.
        
        # 
        if 'route_to_diabetes_model' in str(router_result):
            print("🔬 مرحله 2: دیابت تشخیص داده شد. در حال روتینگ به Agent تخصصی دیابت...")
            
            # 
            final_prediction_output = diabetes_analyst_agent.run(
                prompt="لطفاً تمام داده‌های مورد نیاز را از این PDF استخراج کنید و پیش‌بینی نهایی را انجام دهید.",
                files=[pdf_file_path]
            )
            
            # 
            return f"\n✨ نتیجه تخصصی دیابت:\n{str(final_prediction_output)}"
            
        elif 'route_to_cancer_model' in str(router_result):
            return "\n🩺 نتیجه روتر: سند مربوط به سرطان عمومی است. (نیاز به پیاده‌سازی مدل تخصصی سرطان)"
        
        elif 'route_to_breast_cancer_model' in str(router_result):
            return "\n🩺 نتیجه روتر: سند مربوط به سرطان سینه است. (نیاز به پیاده‌سازی مدل تخصصی سرطان سینه)"

        else: 
            return "\n❌ نتیجه روتر: سند آپلود شده مربوط به دسته‌بندی‌های مورد نظر (دیابت/سرطان) نیست."

    except Exception as e:
        return f"\n❌ خطای کلی در جریان کار: {e}"


# 

if __name__ == '__main__':
    # 
    # 
    test_pdf_path = "diabetes_sample.pdf"
    
    print("\n\n---------------------------------")
    print(f"⚠️ توجه: برای اجرای واقعی، فایل '{test_pdf_path}' باید در مسیر ریشه موجود باشد.")
    print("---------------------------------")
    
    # 
    final_analysis = process_medical_pdf(test_pdf_path)
    
    print("\n\n=================================")
    print("   ✅ تحلیل نهایی:")
    print(final_analysis)
    print("=================================")
