import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import spacy
import re
import io

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_lg")

# Function to get text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Keywords for skills extraction
skills_keywords = ["Python", "Data Analysis", "Machine Learning", "NLP", "Deep Learning", "SQL", "Excel", 
                   "Java", "Spring Boot", "API", "Python", "Javascript", "C++", "Power BI", "Oracle", 
                   "Linux", "Git", "REST API"]

# Common technical terms to exclude from organization extraction
excluded_org_terms = ["Java", "Spring Boot", "API", "Python", "Javascript", "C++", "Power BI", "Oracle", 
                      "Linux", "Git", "REST API", "Institute", "University", "College"]

# Function to parse resume using spaCy and regular expressions, including age, marital status, and education
def parse_resume(text):
    doc = nlp(text)
    
    # Extract Name (Based on PERSON NER tag)
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break  # Assuming first occurrence is the candidate's name

    # Extract Company Name (Based on ORG NER tag and excluding technical terms)
    company_name = []
    for ent in doc.ents:
        if ent.label_ == "ORG" and ent.text not in excluded_org_terms:
            company_name.append(ent.text)

    # Extract Phone Number (Regex for different phone formats)
    phone = re.findall(r'\(?\b[0-9]{3}[-.\)\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', text)
    phone = phone[0] if phone else None  # Taking first found phone number

    # Extract Skills (Based on pre-defined skills list or pattern matching)
    skills = [skill for skill in skills_keywords if skill.lower() in text.lower()]

    # Extract Age (Regex or keyword-based approach)
    age = None
    age_match = re.search(r'Age:?\s?(\d{2})', text, re.IGNORECASE)
    if age_match:
        age = age_match.group(1)

    # Extract Marital Status (Keyword-based approach)
    marital_status = None
    if re.search(r'Marital Status:?\s?(Single|Married|Divorced|Widowed)', text, re.IGNORECASE):
        marital_status = re.search(r'Marital Status:?\s?(Single|Married|Divorced|Widowed)', text, re.IGNORECASE).group(1)

    # Extract Education (Search for education details)
    education = []
    education_patterns = [
        r'B\.\w+ in [\w\s,]+',
        r'M\.\w+ in [\w\s,]+',
        r'Degree in [\w\s,]+',
        r'University of [\w\s,]+',
        r'College of [\w\s,]+'
    ]
    for pattern in education_patterns:
        education.extend(re.findall(pattern, text))

    # Prepare the result dictionary
    parsed_data = {
        "Name": name,
        "Company Name": company_name,
        "Phone": phone,
        "Skills": skills,
        "Age": age,
        "Marital Status": marital_status,
        "Education": education
    }
    return parsed_data

# Function to export parsed resume data to Excel and return it as a downloadable file
def export_to_excel(parsed_data_list):
    # Convert the parsed data list into a pandas DataFrame
    df = pd.DataFrame(parsed_data_list)
    
    # Save the DataFrame to a BytesIO object
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    
    # Seek to the start of the stream
    output.seek(0)
    
    return output

# Modify the Resume Parser section to include handling for multiple resumes
def handle_resume_parsing_and_export(pdf_docs):
    parsed_data_list = []
    
    # Process each uploaded PDF
    for pdf in pdf_docs:
        resume_text = get_pdf_text([pdf])  # Pass the PDF as a list to keep existing get_pdf_text logic
        parsed_data = parse_resume(resume_text)
        parsed_data_list.append(parsed_data)
    
    # Export parsed data to Excel
    excel_data = export_to_excel(parsed_data_list)
    return parsed_data_list, excel_data

# Main function to run the app
def main():
    load_dotenv()
    st.set_page_config(page_title="Resume Parsing", page_icon="📄")
    
    st.title("Resume Parsing Tool")
    
    st.write("Upload multiple resumes in PDF format and generate an Excel file with parsed details.")
    
    # File uploader to handle multiple PDFs
    pdf_docs = st.file_uploader("Upload your resumes in PDF format:", accept_multiple_files=True)
    
    if st.button("Parse and Export Resumes"):
        if pdf_docs:
            with st.spinner("Parsing and Exporting Resumes..."):
                parsed_data_list, excel_data = handle_resume_parsing_and_export(pdf_docs)
                st.write("Parsed Resume Information:")
                st.write(parsed_data_list)
                
                # Provide a download button for the Excel file
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name="parsed_resumes.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("Please upload at least one PDF file.")

if __name__ == '__main__':
    main()
