# gui.py
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, 
    QHBoxLayout, QFrame, QSizePolicy
)
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtCore import Qt
import joblib
import nltk
import re
import sys
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Set base path depending on whether app is frozen (converted to .exe)
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS  # Temporary folder PyInstaller uses
else:
    base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, 'spam_model.pkl')
vectorizer_path = os.path.join(base_path, 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Download stopwords
nltk.download('stopwords')  # Add this before converting
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Prediction function
def predict_sms(sms):
    cleaned = clean_text(sms)
    vec = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vec)[0]
    confidence = model.predict_proba(vec)[0][prediction]
    return prediction, confidence

# GUI Class
class SpamCheckerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📩 SMS Spam Detector")
        self.setGeometry(300, 150, 550, 450)
        self.setStyleSheet(self.get_app_stylesheet())
        self.init_ui()
    
    def get_app_stylesheet(self):
        return """
        QWidget {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        QLabel {
            color: #2c3e50;
        }
        
        QTextEdit {
            background-color: white;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 12px;
            font-size: 11pt;
            color: #2c3e50;
            selection-background-color: #3498db;
        }
        
        QTextEdit:focus {
            border-color: #3498db;
            outline: none;
        }
        
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-size: 10pt;
            font-weight: 500;
            min-width: 75px;
        }
        
        QPushButton:hover {
            background-color: #2980b9;
        }
        
        QPushButton:pressed {
            background-color: #21618c;
        }
        
        QPushButton#clearButton {
            background-color: grey;
        }
        
        QPushButton#clearButton:hover {
            background-color: #7f8c8d;
        }
        
        QPushButton#clearButton:pressed {
            background-color: #6c7b7d;
        }
        
        QFrame#resultFrame {
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0px;
        }
        """
    
    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(25, 25, 25, 25)
        
        # Title
        title_label = QLabel("SMS Spam Detector")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Enter your SMS message below to check if it's spam or legitimate")
        subtitle_label.setFont(QFont("Segoe UI", 10))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #7f8c8d; margin-bottom: 20px;")
        main_layout.addWidget(subtitle_label)
        
        # Input label
        self.input_label = QLabel("Message Content:")
        self.input_label.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.input_label.setStyleSheet("color: #34495e; margin-bottom: 5px;")
        main_layout.addWidget(self.input_label)
        
        # Text input
        self.textbox = QTextEdit()
        self.textbox.setPlaceholderText("Paste your SMS message here...")
        self.textbox.setFixedHeight(120)
        main_layout.addWidget(self.textbox)
        
        # Character counter
        self.char_counter = QLabel("0 characters")
        self.char_counter.setFont(QFont("Segoe UI", 9))
        self.char_counter.setAlignment(Qt.AlignRight)
        self.char_counter.setStyleSheet("color: #95a5a6; margin-top: 5px;")
        self.textbox.textChanged.connect(self.update_char_counter)
        main_layout.addWidget(self.char_counter)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.check_button = QPushButton("🔍 Analyze")
        self.check_button.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.check_button.clicked.connect(self.check_sms)
        button_layout.addWidget(self.check_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("clearButton")
        self.clear_button.setFont(QFont("Segoe UI", 10))
        self.clear_button.clicked.connect(self.clear_text)
        button_layout.addWidget(self.clear_button)
        
        main_layout.addLayout(button_layout)
        
        # Result frame
        self.result_frame = QFrame()
        self.result_frame.setObjectName("resultFrame")
        self.result_frame.setVisible(False)
        result_layout = QVBoxLayout()
        
        self.result_label = QLabel()
        self.result_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        
        self.confidence_label = QLabel()
        self.confidence_label.setFont(QFont("Segoe UI", 10))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("color: #7f8c8d; margin-top: 5px;")
        result_layout.addWidget(self.confidence_label)
        
        self.result_frame.setLayout(result_layout)
        main_layout.addWidget(self.result_frame)
        
        # Add stretch to push everything up
        main_layout.addStretch()
        
        self.setLayout(main_layout)
    
    def update_char_counter(self):
        char_count = len(self.textbox.toPlainText())
        self.char_counter.setText(f"{char_count} characters")
        
        # Change color based on length
        if char_count > 160:
            self.char_counter.setStyleSheet("color: #e67e22; margin-top: 5px;")
        elif char_count > 0:
            self.char_counter.setStyleSheet("color: #27ae60; margin-top: 5px;")
        else:
            self.char_counter.setStyleSheet("color: #95a5a6; margin-top: 5px;")
    
    def check_sms(self):
        message = self.textbox.toPlainText().strip()
        
        if not message:
            self.show_result("⚠️ Please enter a message", "Enter some text to analyze", "#f39c12")
            return
        
        # Disable button during processing
        self.check_button.setText("Analyzing...")
        self.check_button.setEnabled(False)
        
        try:
            pred, conf = predict_sms(message)
            
            if pred == 1:
                self.show_result(
                    "🚫 SPAM DETECTED", 
                    f"Confidence: {conf * 100:.1f}%", 
                    "#e74c3c"
                )
            else:
                self.show_result(
                    "✅ HAM MESSAGE", 
                    f"Confidence: {conf * 100:.1f}%", 
                    "#27ae60"
                )
        
        except Exception as e:
            self.show_result("❌ Error", f"Could not analyze message: {str(e)}", "#e74c3c")
        
        finally:
            self.check_button.setText("🔍 Analyze")
            self.check_button.setEnabled(True)
    
    def show_result(self, main_text, sub_text, color):
        self.result_label.setText(main_text)
        self.result_label.setStyleSheet(f"color: {color};")
        self.confidence_label.setText(sub_text)
        self.result_frame.setVisible(True)
    
    def clear_text(self):
        self.textbox.clear()
        self.result_frame.setVisible(False)
        self.char_counter.setText("0 characters")
        self.char_counter.setStyleSheet("color: #95a5a6; margin-top: 5px;")

# Run the App
if __name__ == "__main__":
    app = QApplication([])
    window = SpamCheckerApp()
    window.show()
    app.exec()