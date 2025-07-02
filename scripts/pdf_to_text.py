import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path, txt_path):
    print(f"Trying to open: {pdf_path}")  # Debug print
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    # Clean up common artifacts
    text = text.replace('\r\n', '\n').replace('\xa0', ' ')

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[✓] Text extracted and saved to: {txt_path}")

if __name__ == "__main__":
    pdf_file = os.path.join("data", "raw", "AtomicHabits.pdf")
    txt_output = os.path.join("data", "cleaned", "AtomicHabits.txt")

    os.makedirs(os.path.dirname(txt_output), exist_ok=True)
    extract_text_from_pdf(pdf_file, txt_output)
