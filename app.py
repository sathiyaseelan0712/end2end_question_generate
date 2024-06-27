import os
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import streamlit as st
from pipeline import E2EQGPipeline

def save_to_docx(questions, doc_filename):
    doc = Document()
    doc.add_heading('Questions', level=1)
    for i, question in enumerate(questions, 1):
        doc.add_paragraph(f"Q{i}: {question}")
    doc.save(doc_filename)
    return doc_filename

def save_to_pdf(questions, pdf_filename):
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.drawString(72, 800, 'Questions')
    y = 780
    for i, question in enumerate(questions, 1):
        c.drawString(72, y - i * 20, f"Q{i}: {question}")
    c.save()
    return pdf_filename

def main():
    st.title("End-to-End Question Generation")
    st.write("Generate questions from your text using a pretrained NLP model.")
    
    context = st.text_area("Enter your text here:")
    num_questions = st.number_input("Number of questions to generate:", min_value=1, value=10)  # Increase value for more questions

    if st.button("Generate Questions"):
        with st.spinner('Generating questions...'):
            nlp = E2EQGPipeline(model_name="valhalla/t5-base-e2e-qg", tokenizer_name="valhalla/t5-base-e2e-qg", use_cuda=False)
            questions = nlp(context, num_questions)
            
            if not questions:
                st.warning("No questions generated. Please try with a different text.")
                return

            # Remove duplicates by converting list to set and back to list
            unique_questions = list(set(questions))

            # Ensure we have enough unique questions
            if len(unique_questions) < num_questions:
                st.warning(f"Only {len(unique_questions)} unique questions generated, instead of {num_questions}.")
                return

            os.makedirs('Questions_docx', exist_ok=True)
            os.makedirs('Questions_pdf', exist_ok=True)
            
            doc_filename = save_to_docx(unique_questions[:num_questions], 'Questions_docx/questions_generated.docx')
            pdf_filename = save_to_pdf(unique_questions[:num_questions], 'Questions_pdf/questions_generated.pdf')

            st.success("Questions generated successfully!")
            
            st.write("### Questions")
            for i, question in enumerate(unique_questions[:num_questions], 1):
                st.write(f"Q{i}: {question}")

            st.write("### Download Links")
            st.markdown(f"[Download Word Document](./{doc_filename})")
            st.markdown(f"[Download PDF Document](./{pdf_filename})")

if __name__ == "__main__":
    main()
