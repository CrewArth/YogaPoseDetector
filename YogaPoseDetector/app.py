from flask import Flask, render_template, request
from pose_correction import PoseCorrection
import cv2
import os

app = Flask(__name__)
pose_correction = PoseCorrection()

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    corrected_image, feedback = pose_correction.detect_pose(image)

    output_path = os.path.join('static', 'output.jpg')
    cv2.imwrite(output_path, corrected_image)

    return render_template('index.html', feedback=feedback, output_image=output_path)

if __name__ == '__main__':
    app.run(debug=True)
