from keras.preprocessing.image import img_to_array
from flask import Flask, render_template, Response,request,session,flash,url_for,redirect,jsonify
import cv2
import numpy as np
from flask_pymongo import PyMongo
from passlib.hash import pbkdf2_sha256
import os
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import librosa
import wave
import pyaudio
import logging
from logging.handlers import RotatingFileHandler


app = Flask(__name__)

app.config['SECRET_KEY'] = '123456'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/your_database_name'

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)


emotion_classifier = load_model("data_mini_XCEPTION.106-0.65.hdf5", compile=False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
Emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

model = load_model('mymodel.h5')
wav_filepath=r"C:\Users\ELDERGOD MAN 24\PycharmProjects\pythonProject8\EMAN\recorded_audio.wav"
output_filename = 'recorded_audio.wav'

detector = MTCNN()

genderProto = "gender_deploy.prototxt" #structure of the model
genderModel = "gender_net.caffemodel" #gender prediction model

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
genderNet = cv2.dnn.readNet(genderModel, genderProto)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        label = Emotions[preds.argmax()]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
        app.logger.info(f"Detected emotion: {label}")
    return frame

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            frame_with_faces = detect_faces(frame)
            ret, buffer = cv2.imencode('.jpg', frame_with_faces)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def detect_gen(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{gender}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
    return frame

def gender_frames():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            frame_with_faces = detect_gen(frame)
            ret, buffer = cv2.imencode('.jpg', frame_with_faces)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def extract_mfcc(wav_file_name):
    # This function extracts mfcc features and obtain the mean of each dimension
    # Input : path_to_wav_file
    # Output: mfcc_features'''
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    return mfccs

def predict(model, wav_filepath):
    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'angry', 5: 'sad', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
    test_point = extract_mfcc(wav_filepath)
    test_point = np.reshape(test_point, newshape=(1, 40, 1))
    predictions = model.predict(test_point)
    text=emotions[np.argmax(predictions[0]) + 1]

    return text

def record_audio(output_filename, duration=5, sample_rate=44100, chunk_size=1024, audio_format=pyaudio.paInt16, channels=1):
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open microphone stream
    stream = audio.open(format=audio_format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

    print("Recording...")

    frames = []

    # Record audio for the specified duration
    for i in range(int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a .wav file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(audio_format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as {output_filename}")
    #predict(model, wav_filepath)


mongo = PyMongo(app)
login_manager = LoginManager(app)
login_manager.login_view = 'signin'

# Define the User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

# Callback to reload the user object
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user and pbkdf2_sha256.verify(request.form['password'], existing_user['password']):
            user_obj = User(existing_user['_id'])
            login_user(user_obj)
            flash('Login successful!', 'success')
            return redirect(url_for('dash'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')

    return render_template('signin.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user is None:
            hashed_password = pbkdf2_sha256.hash(request.form['password'])
            user_id = users.insert_one({'username': request.form['username'], 'password': hashed_password,'email':request.form['email']}).inserted_id
            user_obj = User(user_id)
            login_user(user_obj)
            flash('Account created successfully!', 'success')
            return redirect(url_for('signin'))
        else:
            flash('Username already exists. Please choose a different one.', 'danger')

    return render_template("signup.html")


@app.route('/dash')
@login_required
def dash():
    return render_template('dash.html', username=current_user.id)

@app.route('/eman')
def eman():
    return render_template('eman.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')

        data = {
            'name': name,
            'email': email,
            'subject': subject,
            'message': message
        }
        try:
            mongo.db.feedback_report.insert_one(data)
            flash("Feedback Submitted Successfully", 'success')
        except Exception as e:
            flash("Error submitting form", 'danger')

        return redirect(url_for('dash'))

    # If it's a GET request, you might want to render a template or return a response.
    return render_template('feedback.html')  

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gender_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text', methods=['POST'])
def get_text():
    # Here you can generate or fetch the text dynamically
    record_audio(output_filename)
    text=predict(model, output_filename)
    print(text)
    return jsonify({'text': text})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)