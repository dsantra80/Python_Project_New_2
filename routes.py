from flask import request, jsonify
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

def configure_routes(app):
    model_path = app.config['MODEL_PATH']
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map='auto',
    )

    @app.route('/generate', methods=['POST'])
    def generate():
        data = request.get_json()
        prompt = data.get('prompt')
        max_tokens = app.config['MAX_TOKENS']
        temperature = app.config['TEMPERATURE']

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        generation_output = model.generate(
            input_ids=input_ids, max_new_tokens=max_tokens, temperature=temperature
        )
        response_text = tokenizer.decode(generation_output[0])

        return jsonify({"response": response_text})
