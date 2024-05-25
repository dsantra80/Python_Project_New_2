from flask import request, jsonify
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def generate_response(messages, max_tokens, temperature):
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][len(prompt):]

def configure_routes(app):
    @app.route('/generate', methods=['POST'])
    def generate():
        data = request.get_json()
        messages = data.get('messages')
        max_tokens = app.config['MAX_TOKENS']
        temperature = app.config['TEMPERATURE']
        
        response_text = generate_response(messages, max_tokens, temperature)
        return jsonify({"response": response_text})
