from datetime import datetime
from flask import Flask, redirect, render_template, request, jsonify, url_for, flash
import os
from werkzeug.utils import secure_filename
from services.document_processing.pdf_extractor import extract_pdf_text
from models.legal_research_model import ask_legal_question
from models.fast_chat_model import ask_legal_question_fast, get_model_info, MODEL_OPTIONS
from services.summarization.summarizer_service import summarize_text
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pymysql
pymysql.install_as_MySQLdb()

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:yesmysql@127.0.0.1:3306/newdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['SECRET_KEY'] = 'acc82c31a1412e564cb11b5b45af32c7c7d7b1892d5d1063'

def clean_response(prompt, response):
    lines = response.splitlines()
    if lines and lines[0].strip().lower() == prompt.strip().lower():
        lines = lines[1:]
    return '\n'.join(lines)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    chats = db.relationship('ChatHistory', backref='owner', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query = db.Column(db.String(500), nullable=False)
    response = db.Column(db.Text, nullable=False)  # Changed to TEXT for unlimited length
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<ChatHistory {self.id}>'

with app.app_context():
    db.create_all()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(request.args.get('next') or url_for('home'))
        else:
            flash('Login failed. Check your credentials and try again.', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password, email=email)
        db.session.add(new_user)
        db.session.commit()
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    if request.method == 'POST':
        user_input = request.form.get('prompt')
        if user_input:
            # Use fast model if enabled, otherwise use original model
            use_fast_model = os.getenv('USE_FAST_CHAT_MODEL', 'true').lower() == 'true'
            
            if use_fast_model:
                # Use the new fast chat model
                response = ask_legal_question_fast(user_input)
            else:
                # Use the original Phi-3 model
                response = ask_legal_question(user_input)

            # Clean repeated prompt line from response
            cleaned_response = clean_response(user_input, response)
            
            # Truncate response if extremely long (safety measure, though TEXT column can handle it)
            max_response_length = 50000  # 50KB should be more than enough
            if len(cleaned_response) > max_response_length:
                cleaned_response = cleaned_response[:max_response_length] + "... [Response truncated]"

            # Save chat history with cleaned response
            new_chat = ChatHistory(query=user_input, response=cleaned_response, user_id=current_user.id)
            db.session.add(new_chat)
            db.session.commit()

            # Return cleaned response to frontend
            return jsonify({'response': cleaned_response})

    # For GET requests, render chat UI
    return render_template('chat.html')

@app.route('/summary')
@login_required
def summary():
    return render_template('summary.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        flash('No file part.', 'danger')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'warning')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        extracted_text = extract_pdf_text(filepath)
        
        if not extracted_text or len(extracted_text.strip()) == 0:
            flash('The extracted text is empty.', 'danger')
            return render_template('summary.html', error="The extracted text is empty.")
        
        summary_text = summarize_text(extracted_text)
        
        return render_template('summary.html', extracted_text=extracted_text, summary_text=summary_text)
    
    flash('Invalid file type.', 'danger')
    return redirect(request.url)

@app.route('/api/chat', methods=['POST'])
@login_required
def chat_api():
    user_input = request.json.get('message')
    if user_input:
        # Use fast model if enabled
        use_fast_model = os.getenv('USE_FAST_CHAT_MODEL', 'true').lower() == 'true'
        
        if use_fast_model:
            response = ask_legal_question_fast(user_input)
        else:
            response = ask_legal_question(user_input)
        
        # Truncate response if extremely long (safety measure)
        max_response_length = 50000  # 50KB should be more than enough
        if len(response) > max_response_length:
            response = response[:max_response_length] + "... [Response truncated]"

        new_chat = ChatHistory(query=user_input, response=response, user_id=current_user.id)
        db.session.add(new_chat)
        db.session.commit()

        return jsonify({'response': response})

    return jsonify({'response': 'No input provided.'}), 400

@app.route('/api/chat/model-info', methods=['GET'])
@login_required
def get_chat_model_info():
    """Get information about the currently active chat model."""
    use_fast_model = os.getenv('USE_FAST_CHAT_MODEL', 'true').lower() == 'true'
    
    if use_fast_model:
        model_info = get_model_info()
        model_info['model_type'] = 'fast'
        model_info['available_models'] = {k: v['description'] for k, v in MODEL_OPTIONS.items()}
    else:
        model_info = {
            'model_type': 'original',
            'model_name': 'microsoft/Phi-3.5-mini-instruct',
            'description': 'Original Phi-3.5 model (slower but more capable)',
            'status': 'loaded'
        }
    
    return jsonify(model_info)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False, host='0.0.0.0', port=5000)
