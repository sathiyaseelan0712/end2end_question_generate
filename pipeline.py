import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time

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
            "max_length": 512,          # Increased max_length to accommodate longer generated texts
            "num_beams": 8,             # Increased num_beams for more diverse outputs
            "length_penalty": 1.2,      # Adjusted length_penalty for diversity
            "no_repeat_ngram_size": 3,  # Increased no_repeat_ngram_size for more varied sequences
            "early_stopping": True,
        }

    def __call__(self, context: str, num_questions: int, **generate_kwargs):
        start_time = time.time()
        
        inputs = self._prepare_inputs_for_e2e_qg(context)
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs

        generated_questions = []
        seen_questions = set()

        while len(generated_questions) < num_questions * 2 and time.time() - start_time < 60:  # Adjust time limit as needed
            outs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                num_return_sequences=min(num_questions * 2 - len(generated_questions), 8),  # Increased num_return_sequences
                **generate_kwargs
            )

            for out in outs:
                prediction = self.tokenizer.decode(out, skip_special_tokens=True)
                new_questions = [q.strip() for q in prediction.split("<sep>") if q.strip()]

                for question in new_questions:
                    if question not in seen_questions:
                        seen_questions.add(question)
                        generated_questions.append(question)

                    if len(generated_questions) >= num_questions:
                        break

                if len(generated_questions) >= num_questions:
                    break

        # Ensure we have exactly num_questions unique questions
        questions = list(seen_questions)[:num_questions]

        # If still not enough unique questions, add some random selections
        while len(questions) < num_questions:
            random_question = self._generate_random_question()
            if random_question not in seen_questions:
                questions.append(random_question)
                seen_questions.add(random_question)

        return questions[:num_questions]

    def _generate_random_question(self):
        return "What is the impact of AI on healthcare?"  # Replace with an actual random question generation method

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
