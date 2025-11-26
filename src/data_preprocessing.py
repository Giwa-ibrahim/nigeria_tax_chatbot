import os
import shutil
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

def extract_text_from_doc(doc_path):
    """Extract text from a DOC file (older Word format) using win32com."""
    import win32com.client
    import pythoncom
    
    # Initialize COM for this thread
    pythoncom.CoInitialize()
    
    try:
        # Create Word application instance
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        
        # Open the document
        doc = word.Documents.Open(str(Path(doc_path).absolute()))
        
        # Extract text
        text = doc.Content.Text
        
        # Close document and Word application
        doc.Close(False)
        word.Quit()
        
        return text
    except Exception as e:
        logging.error(f"Error extracting text from DOC file using win32com: {str(e)}")
        raise
    finally:
        # Uninitialize COM
        pythoncom.CoUninitialize()

def move_processed_files(source_folder: str, destination_folder: str):
    """
    Move all files from source folder to destination folder.
    Create destination folder if it doesn't exist.
    """
    source_path = Path(source_folder)
    dest_path = Path(destination_folder)
    
    # Create destination folder if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Move all files from source to destination
    moved_count = 0
    for file_path in source_path.iterdir():
        if file_path.is_file():
            dest_file = dest_path / file_path.name
            
            # If file exists in destination, skip it
            if dest_file.exists():
                logging.info(f"File already exists in destination, skipping: {dest_file.name}")
                continue
            
            shutil.move(str(file_path), str(dest_file))
            logging.info(f"Moved: {file_path.name} -> {dest_file}")
            moved_count += 1
    
    logging.info(f"Total files moved: {moved_count}")
    return moved_count


def process_documents(raw_folder: str, processed_folder: str, used_files_folder: str = None):
    """
    Extract text from PDF, DOCX, and DOC files in raw folder.
    Save each as separate .txt file in processed folder.
    Optionally move processed files to used_files_folder and cleanup.
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
                    
                elif file_path.suffix.lower() == '.doc':
                    text = extract_text_from_doc(file_path)
                    logging.info(f"Processed DOC: {file_path.name}")
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
    
    # Move processed files to used_files folder if specified
    if used_files_folder:
        logging.info(f"\n{'='*10} Moving processed files to {used_files_folder} {'='*10}")
        move_processed_files(raw_folder, used_files_folder)
        
if __name__ == "__main__":
    # Process both subfolders
    subfolders = ["paye_calc", "tax_policy"]
    
    for files in subfolders:
        raw_folder = f"dataset/raw_data/new_raw_files/{files}"
        processed_folder = f"dataset/processed_data/{files}"
        used_files_folder = f"dataset/raw_data/used_files/{files}"
        
        # Check if the raw folder exists before processing
        if Path(raw_folder).exists():
            logging.info(f"\n{'#'*50}")
            logging.info(f"Processing subfolder: {files}")
            logging.info(f"{'#'*50}\n")
            process_documents(raw_folder, processed_folder, used_files_folder)
        else:
            logging.warning(f"Folder does not exist: {raw_folder}")