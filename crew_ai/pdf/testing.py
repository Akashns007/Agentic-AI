from pdfminer.high_level import extract_text

pdf_path = "story.pdf"
extracted_text = extract_text(pdf_path)
print(extracted_text)
