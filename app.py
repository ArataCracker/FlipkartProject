# app.py
import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
import tensorflow as tf
from ultralytics import YOLO
import base64
from datetime import datetime
import cv2
import numpy as np
from pytesseract import pytesseract
import re
from functools import wraps

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

class Result(db.Model):
    __tablename__ = 'result'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model_name = db.Column(db.String(150), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    output = db.Column(db.Text, nullable=False)
    image_data = db.Column(db.LargeBinary, nullable=False)
    image_mimetype = db.Column(db.String(50), nullable=False)

# Load models
freshness_model = tf.keras.models.load_model(os.getenv('FRESHNESS_MODEL_PATH'))
object_detection_model = YOLO(os.getenv('YOLO_MODEL_PATH'))
pytesseract.tesseract_cmd = os.path.join('models', 'tesseract', 'tesseract.exe')

# Helper Decorator

def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash("You need to log in to access this page.", "danger")
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return wrapper

# Helper Functions
def save_result_to_db(user_id, model_name, output, image, mimetype):
    new_result = Result(
        user_id=user_id,
        model_name=model_name,
        output=str(output),
        image_data=image,
        image_mimetype=mimetype
    )
    db.session.add(new_result)
    db.session.commit()

def preprocess_image(image_data, size=(64, 64)):
    np_image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    # Check if the image is loaded correctly
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image.")
    
    image = cv2.resize(image, size)  # Resize the image
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)


def extract_dates_with_keywords(text):
    keywords = ["expiry", "best before", "mfg", "use by"]
    date_pattern = r'\d{2}/\d{2}/\d{4}'
    matches = re.findall(date_pattern, text)
    return matches

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        if User.query.filter_by(email=email).first() or User.query.filter_by(username=username).first():
            flash('Username or email already exists.', 'danger')
            return redirect(url_for('signup'))

        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))

        flash('Invalid email or password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/models/<model_name>', methods=['GET', 'POST'])
@login_required
def models(model_name):
    if request.method == 'POST':
        image_data_url = request.form.get('captured_image')
        if not image_data_url:
            flash('No image captured.', 'danger')
            return redirect(url_for('dashboard'))

        # Decode the base64 image
        header, encoded = image_data_url.split(',', 1)
        image_data = base64.b64decode(encoded)
        mimetype = header.split(':')[1].split(';')[0]

        # Process the image
        try:
            processed_image = preprocess_image(image_data)
            # Continue with your model processing...
        except ValueError as e:
            flash(str(e), 'danger')
            return redirect(url_for('dashboard'))

    return render_template('capture.html', model_name=model_name)

@app.route('/results')
@login_required
def results():
    user_id = session['user_id']
    user_results = Result.query.filter_by(user_id=user_id).all()
    return render_template('results.html', results=user_results)

@app.route('/results/image/<int:result_id>')
@login_required
def serve_image(result_id):
    result = Result.query.get(result_id)
    if not result or result.user_id != session['user_id']:
        flash('You are not authorized to view this image.', 'danger')
        return redirect(url_for('results'))

    return app.response_class(result.image_data, mimetype=result.image_mimetype)

# Model-specific processing

def process_freshness_model(image_data):
    try:
        image = preprocess_image(image_data)
        prediction = freshness_model.predict(image)
        class_index = np.argmax(prediction[0])
        return {"Freshness": class_index}
    except Exception as e:
        return {"Error": str(e)}

def process_object_detection_model(image_data):
    try:
        image = preprocess_image(image_data, size=(640, 640))
        results = object_detection_model(image)
        return results[0].boxes
    except Exception as e:
        return {"Error": str(e)}

def process_expiry_date_model(image_data):
    try:
        text = pytesseract.image_to_string(image_data)
        return extract_dates_with_keywords(text)
    except Exception as e:
        return {"Error": str(e)}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)














# import os
# from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
# from flask_sqlalchemy import SQLAlchemy
# from flask_bcrypt import Bcrypt
# from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
# import cv2
# import numpy as np
# from datetime import datetime
# import tensorflow as tf
# from ultralytics import YOLO
# import pytesseract
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Flask app initialization
# app = Flask(__name__)
# app.secret_key = os.getenv('SECRET_KEY')
# app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
# app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')

# db = SQLAlchemy(app)
# bcrypt = Bcrypt(app)
# jwt = JWTManager(app)

# # Database Models
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(150), unique=True, nullable=False)
#     email = db.Column(db.String(150), unique=True, nullable=False)
#     password = db.Column(db.String(150), nullable=False)

# class Result(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     model_name = db.Column(db.String(50), nullable=False)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     output = db.Column(db.Text, nullable=False)  # Store the result
#     image_data = db.Column(db.LargeBinary, nullable=False)  # Store the image
#     image_mimetype = db.Column(db.String(50), nullable=False)  # Store the image format (e.g., "image/jpeg")


# # Load models
# models_dir = os.path.join(os.getcwd(), "models")
# brand_item_model = YOLO(os.path.join(models_dir, "yolov8n.pt"))
# freshness_model = tf.keras.models.load_model(os.path.join(models_dir, "freshness_model.h5"))
# pytesseract.pytesseract.tesseract_cmd = os.path.join(models_dir, "tesseract", "tesseract.exe")

# # Routes
# @app.route('/')
# def home():
#     if 'user_id' in session:
#         return redirect(url_for('dashboard'))
#     return render_template('login.html')

# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         username = request.form['username']
#         email = request.form['email']
#         password = request.form['password']
#         hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
#         if User.query.filter_by(email=email).first():
#             flash('Email already exists', 'danger')
#             return redirect(url_for('signup'))

#         user = User(username=username, email=email, password=hashed_password)
#         db.session.add(user)
#         db.session.commit()
#         flash('Signup successful! Please login.', 'success')
#         return redirect(url_for('home'))
    
#     return render_template('signup.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']
#         user = User.query.filter_by(email=email).first()

#         if user and bcrypt.check_password_hash(user.password, password):
#             session['user_id'] = user.id
#             session['username'] = user.username
#             return redirect(url_for('dashboard'))
#         else:
#             flash('Invalid credentials. Please try again.', 'danger')
    
#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     session.clear()
#     flash('You have been logged out.', 'success')
#     return redirect(url_for('home'))

# @app.route('/dashboard')
# @jwt_required()
# def dashboard():
#     current_user = get_jwt_identity()
#     results = Result.query.filter_by(user_id=current_user).all()
#     return render_template('dashboard.html', results=results)

# @app.route('/models/<model_name>', methods=['GET', 'POST'])
# @jwt_required()
# def models(model_name):
#     if request.method == 'POST':
#         captured_image_data = request.form['captured_image']
#         nparr = np.frombuffer(captured_image_data, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         try:
#             if model_name == "brand-item":
#                 results = brand_item_model(image)
#                 response = results[0].boxes
#                 response_data = [{"class": brand_item_model.names[int(box.cls)], "confidence": box.conf.item()} for box in response]
#             elif model_name == "freshness":
#                 resized_img = cv2.resize(image, (64, 64))
#                 input_arr = np.expand_dims(resized_img, axis=0)
#                 predictions = freshness_model.predict(input_arr)
#                 response_data = {"class": np.argmax(predictions[0]), "confidence": max(predictions[0])}
#             elif model_name == "expiry":
#                 text = pytesseract.image_to_string(image)
#                 response_data = {"text": text}
#             else:
#                 return jsonify({"error": "Invalid model name"}), 400
#         except Exception as e:
#             return jsonify({"error": "Error processing image", "details": str(e)}), 500

#         result = Result(user_id=session['user_id'], model_name=model_name, result_data=str(response_data))
#         db.session.add(result)
#         db.session.commit()

#         return jsonify({"model_name": model_name, "response_data": response_data})
    
#     return render_template('live_capture.html', model_name=model_name)

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True)





# # import os
# # import secrets
# # from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
# # from flask_sqlalchemy import SQLAlchemy
# # from flask_bcrypt import Bcrypt
# # from flask_jwt_extended import JWTManager, create_access_token, jwt_required
# # from werkzeug.utils import secure_filename
# # from dotenv import load_dotenv
# # import tensorflow as tf
# # from ultralytics import YOLO
# # import cv2
# # import numpy as np
# # from pytesseract import pytesseract
# # from datetime import datetime
# # import re

# # # Load environment variables
# # load_dotenv()

# # # Flask app setup
# # app = Flask(__name__)
# # app.secret_key = secrets.token_hex(16)

# # # Database configuration
# # app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
# # app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# # # JWT configuration
# # app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
# # jwt = JWTManager(app)

# # # Initialize extensions
# # db = SQLAlchemy(app)
# # bcrypt = Bcrypt(app)

# # # Models for database
# # class User(db.Model):
# #     id = db.Column(db.Integer, primary_key=True)
# #     username = db.Column(db.String(150), nullable=False, unique=True)
# #     email = db.Column(db.String(150), nullable=False, unique=True)
# #     password = db.Column(db.String(150), nullable=False)

# # # Load machine learning models
# # freshness_model = tf.keras.models.load_model(os.getenv('FRESHNESS_MODEL_PATH'))
# # object_detection_model = YOLO(os.getenv('YOLO_MODEL_PATH'))

# # # Tesseract configuration
# # pytesseract.tesseract_cmd = os.path.join('models', 'tesseract', 'tesseract.exe')

# # # Routes
# # @app.route('/')
# # def home():
# #     return redirect(url_for('login'))

# # @app.route('/signup', methods=['GET', 'POST'])
# # def signup():
# #     if request.method == 'POST':
# #         username = request.form['username']
# #         email = request.form['email']
# #         password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

# #         if User.query.filter_by(email=email).first() or User.query.filter_by(username=username).first():
# #             flash('Username or email already exists.', 'danger')
# #             return redirect(url_for('signup'))

# #         user = User(username=username, email=email, password=password)
# #         db.session.add(user)
# #         db.session.commit()
# #         flash('Signup successful! Please log in.', 'success')
# #         return redirect(url_for('login'))

# #     return render_template('signup.html')

# # @app.route('/login', methods=['GET', 'POST'])
# # def login():
# #     if request.method == 'POST':
# #         email = request.form['email']
# #         password = request.form['password']

# #         user = User.query.filter_by(email=email).first()
# #         if user and bcrypt.check_password_hash(user.password, password):
# #             access_token = create_access_token(identity=user.id)
# #             session['user_id'] = user.id
# #             flash('Login successful!', 'success')
# #             return redirect(url_for('dashboard'))

# #         flash('Invalid email or password.', 'danger')

# #     return render_template('login.html')

# # @app.route('/logout')
# # def logout():
# #     session.pop('user_id', None)
# #     flash('Logged out successfully.', 'success')
# #     return redirect(url_for('login'))

# # @app.route('/dashboard')
# # @jwt_required()
# # def dashboard():
# #     return render_template('dashboard.html')

# # @app.route('/models/<model_name>', methods=['GET', 'POST'])
# # @jwt_required()
# # def models(model_name):
# #     if request.method == 'POST':
# #         image_data = request.form.get('captured_image')
# #         if not image_data:
# #             flash('No image captured.', 'danger')
# #             return redirect(url_for('dashboard'))

# #         # Process image for each model
# #         if model_name == 'freshness':
# #             result = process_freshness_model(image_data)
# #         elif model_name == 'object_detection':
# #             result = process_object_detection_model(image_data)
# #         elif model_name == 'expiry_date':
# #             result = process_expiry_date_model(image_data)
# #         else:
# #             flash('Invalid model selection.', 'danger')
# #             return redirect(url_for('dashboard'))

# #         return render_template('result.html', result=result)

# #     return render_template('capture.html', model_name=model_name)

# # # Model-specific functions
# # def process_freshness_model(image_data):
# #     try:
# #         image = preprocess_image(image_data)
# #         prediction = freshness_model.predict(image)
# #         class_index = np.argmax(prediction[0])
# #         return {"Freshness": class_index}
# #     except Exception as e:
# #         return {"Error": str(e)}

# # def process_object_detection_model(image_data):
# #     try:
# #         image = preprocess_image(image_data, size=(640, 640))
# #         results = object_detection_model(image)
# #         return results[0].boxes
# #     except Exception as e:
# #         return {"Error": str(e)}

# # def process_expiry_date_model(image_data):
# #     try:
# #         text = pytesseract.image_to_string(image_data)
# #         return extract_dates_with_keywords(text)
# #     except Exception as e:
# #         return {"Error": str(e)}

# # def preprocess_image(image_data, size=(64, 64)):
# #     np_image = np.frombuffer(image_data, np.uint8)
# #     image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
# #     image = cv2.resize(image, size)
# #     image = image / 255.0  # Normalize
# #     return np.expand_dims(image, axis=0)

# # def extract_dates_with_keywords(text):
# #     keywords = ["expiry", "best before", "mfg", "use by"]
# #     date_pattern = r'\d{2}/\d{2}/\d{4}'
# #     matches = re.findall(date_pattern, text)
# #     return matches

# # if __name__ == '__main__':
# #     app.run(debug=True)


# # import os
# # import secrets
# # from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
# # from flask_sqlalchemy import SQLAlchemy
# # from flask_bcrypt import Bcrypt
# # from werkzeug.utils import secure_filename
# # from dotenv import load_dotenv
# # import tensorflow as tf
# # from ultralytics import YOLO
# # import cv2
# # import numpy as np
# # from pytesseract import pytesseract
# # from datetime import datetime
# # import re
# # from functools import wraps

# # # Load environment variables
# # load_dotenv()

# # # Flask app setup
# # app = Flask(__name__)
# # app.secret_key = secrets.token_hex(16)

# # # Database configuration
# # app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
# # app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# # # Initialize extensions
# # db = SQLAlchemy(app)
# # bcrypt = Bcrypt(app)

# # # Models for database
# # class User(db.Model):
# #     id = db.Column(db.Integer, primary_key=True)
# #     username = db.Column(db.String(150), nullable=False, unique=True)
# #     email = db.Column(db.String(150), nullable=False, unique=True)
# #     password = db.Column(db.String(150), nullable=False)

# # # Load machine learning models
# # freshness_model = tf.keras.models.load_model(os.getenv('FRESHNESS_MODEL_PATH'))
# # object_detection_model = YOLO(os.getenv('YOLO_MODEL_PATH'))

# # # Tesseract configuration
# # pytesseract.tesseract_cmd = os.path.join('models', 'tesseract', 'tesseract.exe')

# # # Helper decorator for session-based login
# # def login_required(func):
# #     @wraps(func)
# #     def wrapper(*args, **kwargs):
# #         if 'user_id' not in session:
# #             flash("You need to log in to access this page.", "danger")
# #             return redirect(url_for('login'))
# #         return func(*args, **kwargs)
# #     return wrapper

# # # Routes
# # @app.route('/')
# # def home():
# #     return redirect(url_for('login'))

# # @app.route('/signup', methods=['GET', 'POST'])
# # def signup():
# #     if request.method == 'POST':
# #         username = request.form['username']
# #         email = request.form['email']
# #         password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

# #         if User.query.filter_by(email=email).first() or User.query.filter_by(username=username).first():
# #             flash('Username or email already exists.', 'danger')
# #             return redirect(url_for('signup'))

# #         user = User(username=username, email=email, password=password)
# #         db.session.add(user)
# #         db.session.commit()
# #         flash('Signup successful! Please log in.', 'success')
# #         return redirect(url_for('login'))

# #     return render_template('signup.html')

# # @app.route('/login', methods=['GET', 'POST'])
# # def login():
# #     if request.method == 'POST':
# #         email = request.form['email']
# #         password = request.form['password']

# #         user = User.query.filter_by(email=email).first()
# #         if user and bcrypt.check_password_hash(user.password, password):
# #             session['user_id'] = user.id
# #             flash('Login successful!', 'success')
# #             return redirect(url_for('dashboard'))

# #         flash('Invalid email or password.', 'danger')

# #     return render_template('login.html')

# # @app.route('/logout')
# # def logout():
# #     session.pop('user_id', None)
# #     flash('Logged out successfully.', 'success')
# #     return redirect(url_for('login'))

# # @app.route('/dashboard')
# # @login_required
# # def dashboard():
# #     return render_template('dashboard.html')

# # @app.route('/models/<model_name>', methods=['GET', 'POST'])
# # @login_required
# # def models(model_name):
# #     if request.method == 'POST':
# #         image_data = request.form.get('captured_image')
# #         if not image_data:
# #             flash('No image captured.', 'danger')
# #             return redirect(url_for('dashboard'))

# #         # Process image for each model
# #         if model_name == 'freshness':
# #             result = process_freshness_model(image_data)
# #         elif model_name == 'object_detection':
# #             result = process_object_detection_model(image_data)
# #         elif model_name == 'expiry_date':
# #             result = process_expiry_date_model(image_data)
# #         else:
# #             flash('Invalid model selection.', 'danger')
# #             return redirect(url_for('dashboard'))

# #         return render_template('result.html', result=result)

# #     return render_template('capture.html', model_name=model_name)

# # # Model-specific functions
# # def process_freshness_model(image_data):
# #     try:
# #         image = preprocess_image(image_data)
# #         prediction = freshness_model.predict(image)
# #         class_index = np.argmax(prediction[0])
# #         return {"Freshness": class_index}
# #     except Exception as e:
# #         return {"Error": str(e)}

# # def process_object_detection_model(image_data):
# #     try:
# #         image = preprocess_image(image_data, size=(640, 640))
# #         results = object_detection_model(image)
# #         return results[0].boxes
# #     except Exception as e:
# #         return {"Error": str(e)}

# # def process_expiry_date_model(image_data):
# #     try:
# #         text = pytesseract.image_to_string(image_data)
# #         return extract_dates_with_keywords(text)
# #     except Exception as e:
# #         return {"Error": str(e)}

# # def preprocess_image(image_data, size=(64, 64)):
# #     np_image = np.frombuffer(image_data, np.uint8)
# #     image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
# #     image = cv2.resize(image, size)
# #     image = image / 255.0  # Normalize
# #     return np.expand_dims(image, axis=0)

# # def extract_dates_with_keywords(text):
# #     keywords = ["expiry", "best before", "mfg", "use by"]
# #     date_pattern = r'\d{2}/\d{2}/\d{4}'
# #     matches = re.findall(date_pattern, text)
# #     return matches

# # if __name__ == '__main__':
# #     app.run(debug=True)

