import os
import shutil
import streamlit as st
import pdfplumber
from pathlib import Path
from transformers import pipeline
import tempfile
import zipfile

# Cache model loading to speed up repeat runs and use lightweight model
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

keyword_extractor = load_model()
AI_CATEGORIES = ["Resume", "Bill", "Invoice", "Assignment", "Notes", "Project", "Certificate"]

FILE_TYPES = {
    "Documents": [".pdf", ".docx", ".txt"],
    "Images": [".jpg", ".jpeg", ".png", ".gif"],
    "Archives": [".zip", ".rar", ".tar", ".gz"],
    "Media": [".mp4", ".mp3", ".mkv"],
    "Code": [".py", ".cpp", ".js", ".html", ".css"]
}

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_pdf_text(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except:
        return ""

def extract_txt_text(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""

def ai_classify(text):
    if not text.strip():
        return None
    result = keyword_extractor(text[:1000], candidate_labels=AI_CATEGORIES)
    return result["labels"][0] if result["scores"][0] > 0.5 else None

def organize_files(upload_folder):
    summary = {}
    folder_path = Path(upload_folder)

    for file in folder_path.iterdir():
        if file.is_file():
            ext = file.suffix.lower()
            moved = False

            if ext in [".pdf", ".txt"]:
                text = extract_pdf_text(file) if ext == ".pdf" else extract_txt_text(file)
                category = ai_classify(text)
                if category:
                    target = folder_path / category
                    create_folder(target)
                    shutil.move(str(file), str(target / file.name))
                    summary[category] = summary.get(category, 0) + 1
                    continue

            for category, extensions in FILE_TYPES.items():
                if ext in extensions:
                    target = folder_path / category
                    create_folder(target)
                    shutil.move(str(file), str(target / file.name))
                    summary[category] = summary.get(category, 0) + 1
                    moved = True
                    break

            if not moved:
                other = folder_path / "Others"
                create_folder(other)
                shutil.move(str(file), str(other / file.name))
                summary["Others"] = summary.get("Others", 0) + 1

    return summary

def zip_folder(folder_path):
    zip_path = folder_path + ".zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    return zip_path

def display_folder_structure(folder_path):
    for root, dirs, files in os.walk(folder_path):
        level = root.replace(folder_path, '').count(os.sep)
        if level == 0:
            continue  # Skip temp root folder name
        indent = '    ' * level
        st.markdown(f"{indent}📁 **{os.path.basename(root)}**")
        sub_indent = '    ' * (level + 1)
        for file in files:
            st.markdown(f"{sub_indent}- {file}")

st.set_page_config(page_title="Smart File Organizer", page_icon="📁")
st.title("📁 Smart File Organizer")

uploaded_files = st.file_uploader("Upload multiple files to organize:", type=None, accept_multiple_files=True)

if uploaded_files:
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = []
        for uploaded_file in uploaded_files:
            if uploaded_file.size > 10 * 1024 * 1024:  # Skip large files
                st.warning(f"⚠️ {uploaded_file.name} is too large and was skipped.")
                continue

            file_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            file_paths.append(file_path)

        if st.button("Organize Files"):
            with st.spinner("Organizing uploaded files..."):
                result = organize_files(tmp_dir)
                zip_file = zip_folder(tmp_dir)

            st.success("✅ Done! Here's what was organized:")
            for category, count in result.items():
                st.write(f"📂 {category}: {count} file(s)")

            st.markdown("---")
            st.subheader("📂 Folder Preview:")
            display_folder_structure(tmp_dir)

            with open(zip_file, "rb") as f:
                st.download_button("📦 Download Organized Files as ZIP", f, file_name="organized_files.zip")
