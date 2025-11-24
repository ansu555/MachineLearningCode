import os
from pypdf import PdfReader

def extract_text_from_pdfs(pdf_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_filename)
            
            print(f"Extracting {filename}...")
            try:
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Saved to {txt_path}")
            except Exception as e:
                print(f"Failed to extract {filename}: {e}")

if __name__ == "__main__":
    extract_text_from_pdfs('pdfs', 'extracted_text')
