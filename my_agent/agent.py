# =================================================================
# فایل: my_agent/agent.py
# =================================================================
import pickle
import pandas as pd
import os
from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'final_diabetes_model.pkl')

LOADED_MODEL = None

MODEL_EXPECTED_FEATURES: List[str] = [
    'GenderFemale1Male2', 'Ageyears', 'Heightm', 'Weightkg', 'BMIkgm2', 
    'SmokingHistorypackyear', 'AlcoholDrinkingHistorydrinkernondrinker', 
    'Durationofdiabetesyears', 'has_PADperipheralarterialdisease', 
    'has_CHDcoronaryheartdisease', 'nephropathy', 'retinopathy', 
    'neuropathy', '2hourPostprandialPlasmaGlucosemgdl', 'FastingCpeptidenmolL', 
    '2hourPostprandialCpeptidenmolL', 'FastingInsulinpmolL', 
    '2hourPostprandialinsulinpmolL', 'HbA1cmmolmol', 'GlycatedAlbumin', 
    'TotalCholesterolmmolL', 'TriglyceridemmolL', 'HighDensityLipoproteinCholesterolmmolL', 
    'LowDensityLipoproteinCholesterolmmolL', 'CreatinineumolL', 
    'EstimatedGlomerularFiltrationRatemlmin173m2', 'UricAcidmmolL', 
    'BloodUreaNitrogenmmolL', 'Hypoglycemiayesno', 'hypertension', 
    'hyperuricemia', 'nephrolithiasis', 'kidneycyst', 'prostatichyperplasia', 
    'hepaticdysfunction', 'chronichepatitisB', 'vitaminDdeficiency', 
    'atrialfibrillation', 'pulmonarynodule', 'thyroidnodule', 
    'has_CVDcerebrovasculardisease', 'Alzheimersdisease', 
    'urinarytractinfection', 'anxiety', 'hyperlipidemia', 'fattyliverdisease', 
    'hypokalemiaCustom', 'conjunctivitis', 'osteoporosis', 'cataract', 
    'enlargedadrenalgland', 'gallbladderpolyp', 'livercyst', 'cholelithiasis', 
    'Parkinsonsdisease', 'pancreaticcancer', 'osteopenia', 
    'lumbarherniateddisc', 'periodontitis', 'chronicatrophicgastritis', 
    'hydronephrosis', 'chronicgastritis', 'gastricpolyp', 'colorectalpolyp', 
    'sinusbradycardia', 'sinusarrhythmia', 'hypothyroidism', 'hysteromyoma', 
    'cholecystitis', 'hypoparathyroidism', 'psoriasis', 'myocardialbridging', 
    'parotidglandcarcinoma', 'lunglesion', 'Drug_metformin', 'Drug_Humulin7030', 
    'Drug_insulinaspart5050', 'Drug_acarbose', 'Drug_glimepiride', 
    'Drug_insulinaspart7030', 'Drug_voglibose', 'Drug_Novolin30R', 
    'Drug_NovolinR', 'Drug_pioglitazone', 'Drug_sitagliptin', 
    'Drug_gliclazide', 'Drug_liraglutide', 'Drug_insulindegludec', 
    'Drug_insulinglarigine', 'Drug_insulindetemir', 'Drug_insulinglulisine', 
    'Drug_HumulinR', 'Drug_Gansulin40R', 'Drug_GansulinR', 'Drug_gliquidone', 
    'Drug_canagliflozin', 'Drug_dapagliflozin', 'Drug_repaglinide', 
    'Other_drug_atorvastatin', 'Other_drug_aspirin', 'Other_drug_nifedipine', 
    'Other_drug_calcium_dobesilate', 'Other_drug_epalrestat', 
    'Other_drug_mecobalamin', 'Other_drug_calcitriol', 
    'Other_drug_polyene_phosphatidylcholine', 'Other_drug_olmesartan', 
    'Other_drug_metoprolol', 'Other_drug_vitamin_B1', 'Other_drug_clopidogrel', 
    'Other_drug_bisoprolol', 'Other_drug_amlodipine', 'Other_drug_valsartan', 
    'Other_drug_rabeprazole', 'Other_drug_febuxostat', 
    'Other_drug_calcium_carbonate_vitD3', 'Other_drug_pravastatin', 
    'Other_drug_doxazosin', 'Other_drug_pancreatic_kininogenase', 
    'Other_drug_rosuvastatin', 'Other_drug_irbesartan', 
    'Other_drug_benazepril', 'Other_drug_candesartan', 'Other_drug_felodipine', 
    'Other_drug_telmisartan', 'Other_drug_losartan', 'Other_drug_losartan_HCTZ', 
    'Other_drug_benidipine', 'Other_drug_allisartan', 'Other_drug_labetalol', 
    'Other_drug_levofloxacin', 'Other_drug_rivaroxaban', 'Other_drug_quetiapine', 
    'Other_drug_betahistine', 'Other_drug_bisacodyl', 
    'Other_drug_clostridium_butyricum', 'Other_drug_fenofibrate', 
    'Other_drug_ezetimibe', 'Other_drug_levothyroxine', 
    'Other_drug_magnesium_isoglycyrrhizinate', 'Other_drug_multivitamin', 
    'Other_drug_beiprostaglandin_sodium', 'Other_drug_compound_alpha_keto_acid', 
    'Other_drug_potassium_chloride', 'Other_drug_Zhenju_Jiangya_tablet', 
    'Other_drug_Yinxingye_tablet', 'Other_drug_Qianlie_Shutong_capsule', 
    'Other_drug_Shen_Shuai_Ning_capsule'
]

# 
try:
    with open(MODEL_PATH, 'rb') as file:
        LOADED_MODEL = pickle.load(file)
    # print("✅ مدل پیش‌بینی داده‌های دیابت با موفقیت بارگذاری شد.") # این پرینت در ADK اجرا می‌شود
except FileNotFoundError:
    print(f"❌ خطا: فایل مدل در مسیر {MODEL_PATH} یافت نشد. Agent نمی‌تواند پیش‌بینی کند.")
except Exception as e:
    print(f"❌ خطای بارگذاری مدل: {e}")


def predict_data_outcome(data_features: Dict[str, Any]) -> str:
    """
    Predicts the Fasting Plasma Glucose (mg/dl) based on the input features.
    
    Args:
        data_features (Dict[str, Any]): A dictionary containing all 
        the necessary features for the prediction model (148 keys).
    """
    if LOADED_MODEL is None:
        return "خطا: مدل یادگیری ماشین در دسترس نیست."

    #  
    input_data = {}
    for feature in MODEL_EXPECTED_FEATURES:
        #
        input_data[feature] = data_features.get(feature, 0)
    
    try:
        #  turning DataFrame to dictionary
        input_df = pd.DataFrame([input_data])
        
        prediction = LOADED_MODEL.predict(input_df)
        
        predicted_value = float(prediction[0])
        return f"مقدار پیش‌بینی شده Fasting Plasma Glucose (mg/dl): {predicted_value:.2f}"
    
    except Exception as e:
        return f"خطا در هنگام پیش‌بینی: مطمئن شوید تمام ویژگی‌های لازم به درستی ارائه شده‌اند. جزئیات خطا: {e}"

# Diabet Agent 
prediction_tool = FunctionTool(func=predict_data_outcome)

# 
diabetes_analyst_agent = Agent(
    model='gemini-2.5-flash',
    name='diabetes_ml_analyst',
    description="A specialized medical data analyst agent that utilizes a pre-trained machine learning model for prediction and data analysis.",
    instruction=(
        "شما یک دستیار تحلیلگر داده پزشکی متخصص دیابت هستید. وظیفه شما این است که "
        "تمام داده‌های بیمار را از PDF آپلود شده استخراج کنید. این داده‌ها شامل 148 ویژگی مورد نیاز مدل یادگیری ماشین است. "
        "از محتوای PDF، مقادیر عددی یا وضعیت‌های بله/خیر (0 یا 1) را برای تمام ویژگی‌های زیر استخراج کنید. "
        "پس از استخراج، ابزار 'predict_data_outcome' را با دیکشنری کامل (شامل تمام 148 کلید) فراخوانی کنید. "
        "برای ویژگی‌هایی که در PDF یافت نمی‌شوند، مقدار پیش‌فرض 0 را در دیکشنری قرار دهید. "
        "فقط خروجی نهایی از ابزار را برگردانید.\n\n"
        f"فهرست کامل ویژگی‌های مورد نیاز: {MODEL_EXPECTED_FEATURES}"
    ),
    tools=[prediction_tool],
)
