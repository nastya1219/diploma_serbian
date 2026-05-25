import os
import PyPDF2

results_dir = 'results'

if not os.path.exists(results_dir):
    print("Results directory not found")
    exit()

for filename in os.listdir(results_dir):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(results_dir, filename)
        txt_filename = filename.replace('.pdf', '.txt')
        txt_path = os.path.join(results_dir, txt_filename)
        try:
            with open(pdf_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
            print(f"Extracted {filename} to {txt_filename}")
        except Exception as e:
            print(f"Error extracting {filename}: {e}")

print("Extraction complete")