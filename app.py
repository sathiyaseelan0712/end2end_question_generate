import os
import torch
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st

class E2EQGPipeline:
    def __init__(self, model_name: str, tokenizer_name: str, use_cuda: bool = False):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]

        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"

        self.default_generate_kwargs = {
            "max_length": 256,
            "num_beams": 20,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

    def __call__(self, context: str, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        questions = [question.strip() for question in questions if question.strip()]
        return questions

    def _prepare_inputs_for_e2e_qg(self, context):
        source_text = f"generate questions: {context}"
        if self.model_type == "t5":
            source_text += " </s>"

        return self._tokenize([source_text], padding=False)

    def _tokenize(self, inputs, padding=True, truncation=True, add_special_tokens=True, max_length=512):
        return self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )

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

    if st.button("Generate Questions"):
        with st.spinner('Generating questions...'):
            nlp = E2EQGPipeline(model_name="valhalla/t5-base-e2e-qg", tokenizer_name="valhalla/t5-base-e2e-qg", use_cuda=False)
            questions = nlp(context)
            
            if not questions:
                st.warning("No questions generated. Please try with a different text.")
                return

            os.makedirs('Questions_docx', exist_ok=True)
            os.makedirs('Questions_pdf', exist_ok=True)
            
            doc_filename = save_to_docx(questions, 'Questions_docx/questions_generated.docx')
            pdf_filename = save_to_pdf(questions, 'Questions_pdf/questions_generated.pdf')

            st.success("Questions generated successfully!")
            
            st.write("### Questions")
            for i, question in enumerate(questions, 1):
                st.write(f"Q{i}: {question}")

            st.write("### Download Links")
            st.markdown(f"[Download Word Document]({doc_filename})")
            st.markdown(f"[Download PDF Document]({pdf_filename})")

if __name__ == "__main__":
    main()
