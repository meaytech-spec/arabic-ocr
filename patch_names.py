# Patch script to fix extract_names_from_full_text function
import re

# Read the file
with open('/app/final_ocr_zones.py', 'r', encoding='utf-8') as f:
    content = f.read()

# New function body
new_function = '''def extract_names_from_full_text(text):
    """Extract names using regex pattern matching"""
    text = clean_text(text)
    
    last_name = ""
    first_name = ""
    
    # Strategy: Names appear after National ID (labels not reliably OCR'd)
    nat_id_match = re.search(r"\\d{15,20}", text)
    
    if nat_id_match:
        text_after_id = text[nat_id_match.end():]
        lines = [l.strip() for l in text_after_id.split('\\n') if l.strip()]
        
        arabic_lines = []
        for line in lines[:5]:
            if re.search(r"الجنس|تاريخ|مكان|\\d{4}\\.\\d{2}\\.\\d{2}", line):
                break
            if re.search(r"[\\u0600-\\u06FF]{2,}", line):
                arabic_text = re.findall(r"[\\u0600-\\u06FF\\s]+", line)
                if arabic_text:
                    cleaned = ' '.join(arabic_text).strip()
                    if len(cleaned) >= 2:
                        arabic_lines.append(cleaned)
        
        if len(arabic_lines) >= 1:
            last_name = arabic_lines[0]
        if len(arabic_lines) >= 2:
            first_name = arabic_lines[1]
    
    # Clean extracted names
    if last_name:
        last_name = clean_name(last_name)
        if not is_valid_arabic_text(last_name):
            last_name = ""
    
    if first_name:
        first_name = clean_name(first_name)
        if not is_valid_arabic_text(first_name):
            first_name = ""
    
    return {
        "last_name": last_name,
        "first_name": first_name
    }'''

# Find and replace the function
pattern = r'def extract_names_from_full_text\(text\):.*?return \{[^}]+\}'
content = re.sub(pattern, new_function, content, flags=re.DOTALL)

# Write back
with open('/app/final_ocr_zones.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Patched extract_names_from_full_text function")
