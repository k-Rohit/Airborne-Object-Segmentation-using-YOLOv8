import math
import os
import cv2
from PIL import Image, ImageDraw
from flask import Flask, render_template, Response
from flask_wtf import FlaskForm
# import requests
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'key1'
app.config['UPLOAD_FOLDER'] = 'Uploads'


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


image_array = []


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def predict():  # put application's code here
    # global results, frame
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data  # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))
        filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                secure_filename(file.filename))
        extension = filepath.rsplit('.', 1)[1].lower()
        if extension == 'jpg':
            img = Image.open(filepath)
            yolo = YOLO('/Users/kumarrohit/PycharmProjects/flaskProject/yolov8n.pt')
            os.chdir(os.getcwd())
            yolo.predict(img, save=True)
        elif extension == 'mp4':

            video_path = filepath

            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('static/output.mp4', fourcc, 30.0, (2448, 2048), True)

            # Get the video dimensions

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Loading the trained model
            model = YOLO('/Users/kumarrohit/PycharmProjects/flaskProject/yolov8n.pt')
            class_names = ['Helicopter', 'Airplane']
            while cap.isOpened():

                success, frame = cap.read()

                if not success:
                    break

                results = model.predict(frame, stream=True)

                for r in results:

                    boxes = r.boxes
                    # using the properties in which we have conf (torch.Tensor) to show the confidence values of the
                    # boxes
                    print(boxes.conf)

                    for box in boxes:
                        x1, y1, x2, y2 = boxes.xyxy[0]
                        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

                        img = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), thickness=4)
                        # using boxes.conf to show the confidence values of the boxes
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        # Class name
                        cls = int(box.cls[0])
                        currentClass = class_names[cls]

                        img = cv2.putText(img, f'{currentClass}{conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                          (0, 255, 255), 2, cv2.LINE_AA)

                        image_array.append(img)
                        # cv2.imwrite()
                        # print(image_array)
                        out.write(img)
            cap.release()
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)),int(cap.get(4))))

        # for i in range(len(image_array)):

        # out.release()

        return render_template('video.html')
    return render_template('index.html', form=form)


predict()
if __name__ == '__main__':
    app.run(debug=True, port=8080)
