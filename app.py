from flask import Flask
from routes import configure_routes

# Initialize the Flask app
app = Flask(__name__)

# Set configuration values
app.config['MODEL_PATH'] = "openlm-research/open_llama_7b"  # Replace with your model path
app.config['MAX_TOKENS'] = 50  # Adjust as needed
app.config['TEMPERATURE'] = 0.7  # Adjust as needed

# Configure routes
configure_routes(app)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
