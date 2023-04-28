import io
import os
import time

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


# def get_frame():
#     folderPath = os.getcwd()
#     file = 'output.mp4'
#     video = cv2.VideoCapture(folderPath + '/' + file)
#     while True:
#         success, image = video.read()
#         if not success:
#             break
#         ret, jpeg = cv2.imencode('.jpg', image)
#         yield jpeg.tobytes()
#         time.sleep(0.1)


# @app.route("/video_path")
# def video_feed():
#     print("Function called")
#     return Response(get_frame(), mimetype='multipart/x-mixed/replace; boundary=frame')


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
            # frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()
            # image = Image.open(io.BytesIO(frame))
            yolo = YOLO('yolov8n.pt')
            os.chdir(os.getcwd())
            detections = yolo.predict(img, save=True)
        elif extension == 'mp4':
            video_path = filepath
            cap = cv2.VideoCapture(video_path)

            # Get the video dimensions

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define the codec and create the video writer object

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('static/output.mp4', fourcc, 30.0, (frame_width, frame_height))

            model = YOLO('yolov8n.pt')
            while cap.isOpened():

                success, frame = cap.read()

                if not success:
                    break

                results = model.predict(frame, stream=True)
                # print(results)
                # cv2.waitKey(1)
                #
                # res_plotted = results[0].plot()
                # cv2.imshow("result", res_plotted)
                #
                # # Write the frame to the output video
                # out.write(res_plotted)

                # image_array.append(frame)

                for r in results:
                    boxes = r.boxes
                    for box in boxes:

                        x1, y1, x2, y2 = boxes.xyxy[0]
                        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                        img = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), thickness=4)
                        image_array.append(img)

                for i in range(len(image_array)):
                    out.write(image_array[i])

        return render_template('video.html')
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True, port=8081)
