from flask import Flask, render_template, send_from_directory, request, url_for
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# Load the model
model = load_model("model/my_model.h5")


def resize_image(filepath):
    img = cv2.imread(filepath)
    image_x = 240
    image_y = 240
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 3))
    return img

def predict_image(filepath):
    img = resize_image(filepath)
    prediction = model.predict(img)
    return "Brain Tumor detected.!!" if prediction == [[1.]] else "No Brain Tumor"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    uploaded_image_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', result="No file part")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', result="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform prediction
            result = predict_image(filepath)
            uploaded_image_path = filename  # Image is now directly in the 'uploads' folder

    return render_template('index.html', result=result, uploaded_image_path=uploaded_image_path)

# Route for the Symptoms page
@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

if __name__ == '__main__':
    app.run(debug=True)
