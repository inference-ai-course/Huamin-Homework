import os
from pdf2image import convert_from_path
import pytesseract
# ✅ 显式指定 Tesseract 安装路径
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
from tqdm import tqdm

# 设置路径
PDF_DIR = "./pdfs"
OUTPUT_DIR = "./pdf_ocr"
POPPLER_PATH = "/Users/jianghm/miniforge3/bin"  # 修改为你的 Poppler 路径

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def ocr_pdf(pdf_path, txt_path):
    try:
        images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
        all_text = ""
        for img in images:
            text = pytesseract.image_to_string(img, lang='eng', config='--psm 1')
            all_text += text + "\n\n"  # 每页后换行
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(all_text)
        return True
    except Exception as e:
        print(f"[Error] Failed on {pdf_path}: {e}")
        return False

def batch_ocr():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    for pdf_file in tqdm(pdf_files, desc="OCR Processing"):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        txt_file = os.path.splitext(pdf_file)[0] + ".txt"
        txt_path = os.path.join(OUTPUT_DIR, txt_file)
        if not os.path.exists(txt_path):  # 跳过已处理
            ocr_pdf(pdf_path, txt_path)

if __name__ == "__main__":
    batch_ocr()
