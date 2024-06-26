import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
            "max_length": 512,  # Increased max_length to accommodate longer generated texts
            "num_beams": 20,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

    def __call__(self, context: str, num_questions: int, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            num_return_sequences=num_questions,  # Generate multiple sequences
            **generate_kwargs
        )

        questions = []
        for out in outs:
            prediction = self.tokenizer.decode(out, skip_special_tokens=True)
            questions += prediction.split("<sep>")
        
        questions = [question.strip() for question in questions if question.strip()]
        return questions[:num_questions]

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
