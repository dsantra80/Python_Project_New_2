from flask import Flask, request, jsonify
import transformers
import torch
import os
import logging

app = Flask(__name__)
app.config['MAX_TOKENS'] = 100  # Example value, set as needed
app.config['TEMPERATURE'] = 0.7  # Example value, set as needed

local_model_path = "./Meta-Llama-3-70B-Instruct"
hf_auth_token = os.getenv('HUGGINGFACE_TOKEN')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Initializing the Hugging Face pipeline...")

    # Use local model if available
    if os.path.exists(local_model_path):
        logger.info(f"Using local model path: {local_model_path}")
        model_id = local_model_path
    else:
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        use_auth_token=hf_auth_token
    )
    logger.info("Pipeline initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize the pipeline: {e}")
    raise

def generate_response(messages, max_tokens, temperature):
    try:
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
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return "Error generating response"

def configure_routes(app):
    @app.route('/generate', methods=['POST'])
    def generate():
        data = request.get_json()
        messages = data.get('messages')
        max_tokens = app.config['MAX_TOKENS']
        temperature = app.config['TEMPERATURE']
        
        response_text = generate_response(messages, max_tokens, temperature)
        return jsonify({"response": response_text})

configure_routes(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

