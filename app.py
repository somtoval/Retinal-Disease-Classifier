from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from predict import predict

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the model once when the app starts
model = tf.keras.models.load_model('retina_model.keras')

@app.route('/')
def home():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = predict(filepath, model)  # Use the loaded model for prediction
            return jsonify(result), 200
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    return jsonify({'error': 'Unsupported file type'}), 400

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
