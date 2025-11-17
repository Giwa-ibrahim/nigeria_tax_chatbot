import os
from pathlib import Path
from pypdf import PdfReader
from docx import Document
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages])
    return text

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    doc = Document(docx_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def process_documents(raw_folder: str, processed_folder: str):
    """
    Extract text from PDF and DOCX files in raw folder.
    Save each as separate .txt file in processed folder.
    """
    raw_path = Path(raw_folder)
    processed_path = Path(processed_folder)
    
    # Create processed folder if it doesn't exist
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Process all files
    logging.info(f"{'='*10} Starting to process files in {raw_folder} {'='*10}")
    for file_path in raw_path.iterdir():
        if file_path.is_file():
            # Normalize file name by converting to lowercase and replacing spaces with underscores
            file_name = file_path.stem.lower().replace(" ", "_")
            output_file = processed_path / f"{file_name}.txt"
            
            try:
                if file_path.suffix.lower() == '.pdf':
                    text = extract_text_from_pdf(file_path)
                    logging.info(f"Processed PDF: {file_path.name}")
                    
                elif file_path.suffix.lower() == '.docx':
                    text = extract_text_from_docx(file_path)
                    logging.info(f"Processed DOCX: {file_path.name}")
                else:
                    logging.info(f"Skipped unsupported file: {file_path.name}")
                    continue
                
                # Write to text file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                    
                logging.info(f"Saved to: {output_file.name}\n")
                
            except Exception as e:
                logging.error(f"Error processing {file_path.name}: {str(e)}")

    logging.info(f"{'='*10} Finished processing files in {raw_folder} {'='*10}")
    
if __name__ == "__main__":
    # denote which folders path to extract from and save to
    raw_folder="dataset/raw_data"
    processed_folder="dataset/processed_data"
    process_documents(raw_folder, processed_folder)