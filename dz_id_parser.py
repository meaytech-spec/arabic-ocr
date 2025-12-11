import re

def extract_cni_data(raw_ocr_text):
    """
    Extracts key information from an Algerian CNI using Arabic keywords as anchors.
    Designed for high-quality OCR output.
    """
    # Clean and normalize the text (replace newlines with spaces)
    text = raw_ocr_text.replace("\n", " ").strip()

    data = {
        "nni": None,
        "full_name_ar": None,
        "family_name_ar": None,
        "given_name_ar": None,
        "gender_ar": None,
        "dob_date": None,
        "dob_place_ar": None,
        "issue_date": None,
        "expiry_date": None,
        "authority_ar": None,
        "national_id_number": None
    }

    # --- NNI (National Identification Number) ---
    # The large number (101463195) is NOT the main NNI, it's the card series number.
    # The NNI is the 18-digit number after 'رقم التعريف الوطني'
    nni_match = re.search(r"رقم التعريف الوطني\s*:\s*(\d{18})", text)
    if nni_match:
        data["national_id_number"] = nni_match.group(1)
    
    # --- Family Name (اللقب) ---
    nom_match = re.search(r"اللقب\s*:\s*(?P<nom>[\u0600-\u06FF\s]+?)(الاسم|الجنس|$)", text)
    if nom_match:
        data["family_name_ar"] = nom_match.group("nom").strip()

    # --- Given Name (الاسم) ---
    prenom_match = re.search(r"الاسم\s*:\s*(?P<prenom>[\u0600-\u06FF\s]+?)(تاريخ الميلاد|مكان الميلاد|$)", text)
    if prenom_match:
        data["given_name_ar"] = prenom_match.group("prenom").strip()
        
    # Combine for Full Name
    if data["family_name_ar"] and data["given_name_ar"]:
        data["full_name_ar"] = f"{data['given_name_ar']} {data['family_name_ar']}"

    # --- Gender (الجنس) ---
    gender_match = re.search(r"الجنس\s*:\s*(?P<gender>[\u0600-\u06FF]+)", text)
    if gender_match:
        data["gender_ar"] = gender_match.group("gender").strip()

    # --- Date of Birth (تاريخ الميلاد) ---
    dob_match = re.search(r"تاريخ الميلاد\s*:\s*(\d{4}\.\d{2}\.\d{2})", text)
    if dob_match:
        data["dob_date"] = dob_match.group(1)

    # --- Place of Birth (مكان الميلاد) ---
    # Captures Arabic text until the next field or end of line
    pob_match = re.search(r"مكان الميلاد\s*:\s*(?P<place>[\u0600-\u06FF\s]+)", text)
    if pob_match:
        data["dob_place_ar"] = pob_match.group("place").strip()

    # --- Issue Authority (سلطة الإصدار) ---
    authority_match = re.search(r"سلطة الإصدار\s*:\s*(?P<auth>[\u0600-\u06FF\s]+?)(تاريخ الإصدار|$)", text)
    if authority_match:
        data["authority_ar"] = authority_match.group("auth").strip()
    
    # --- Issue Date (تاريخ الإصدار) ---
    issue_date_match = re.search(r"تاريخ الإصدار\s*:\s*(\d{4}\.\d{2}\.\d{2})", text)
    if issue_date_match:
        data["issue_date"] = issue_date_match.group(1)

    # --- Expiry Date (تاريخ الإنتهاء) ---
    expiry_date_match = re.search(r"تاريخ الإنتهاء\s*:\s*(\d{4}\.\d{2}\.\d{2})", text)
    if expiry_date_match:
        data["expiry_date"] = expiry_date_match.group(1)

    # Clean up and return
    return {k: v for k, v in data.items() if v is not None}

# --- Simulation of the OCR Output ---
# This simulates the text your chosen OCR engine should produce
simulated_ocr_output = """
الجمهورية الجزائرية الديمقراطية الشعبية بطاقة التعريف الوطنية
101463195
سلطة الإصدار: عين البيضاء-وهران
تاريخ الإصدار: 2016.10.22 تاريخ الإنتهاء: 2026.10.21
رقم التعريف الوطني: 119951110039550003
اللقب: عيواني الاسم: يسرى
الجنس: أنثى تاريخ الميلاد: 1995.04.21
مكان الميلاد: وهران
"""

extracted_info = extract_cni_data(simulated_ocr_output)
print(extracted_info)