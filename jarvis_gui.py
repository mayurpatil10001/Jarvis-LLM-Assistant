import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import speech_recognition as sr
import pyttsx3
import torch
import pickle
import queue
import time
from datetime import datetime
import os
import json

# Import the model classes from training script
import nbimporter
from jarvis_training import JARVISModel, TokenizerVocab

class JARVISAssistant:
    def __init__(self):
        # Initialize speech recognition and TTS
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
        self.setup_tts()
        
        # Load trained model and tokenizer
        self.load_model()
        
        # Threading and state management
        self.is_listening = False
        self.response_queue = queue.Queue()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
    
    def setup_tts(self):
        """Configure text-to-speech engine"""
        voices = self.tts_engine.getProperty('voices')
        
        # Try to set a male voice (JARVIS-like)
        for voice in voices:
            if 'male' in voice.name.lower() or 'david' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        # Set speech rate and volume
        self.tts_engine.setProperty('rate', 180)  # Speed
        self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    
    def load_model(self):
        """Load the trained JARVIS model"""
        try:
            # Load tokenizer
            with open('tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Load model
            checkpoint = torch.load('jarvis_model.pth', map_location='cpu')
            self.model = JARVISModel(vocab_size=checkpoint['vocab_size'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print("Model loaded successfully!")
            
        except FileNotFoundError:
            messagebox.showerror("Error", "Model files not found! Please train the model first.")
            self.model = None
            self.tokenizer = None
    
    def listen_for_speech(self):
        """Listen for speech input"""
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio)
            return text.lower()
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Sorry, speech recognition service is unavailable."
    
    def speak(self, text):
        """Convert text to speech"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def generate_response(self, user_input):
        """Generate response using the trained model"""
        if self.model is None:
            return "Model not loaded. Please train the model first."
        
        try:
            response = self.model.generate(
                self.tokenizer, 
                user_input, 
                max_length=50, 
                temperature=0.7
            )
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

class JARVISGui:
    def __init__(self, root):
        self.root = root
        self.jarvis = JARVISAssistant()
        # Chat history must be initialized before setup_gui (fixes AttributeError)
        self.chat_history = []
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root.title("JARVIS - AI Assistant")
        self.root.geometry("800x600")
        self.root.configure(bg='#1a1a1a')
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background='#1a1a1a', foreground='#00ff00')
        style.configure('TButton', background='#333333', foreground='#00ff00')
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="J.A.R.V.I.S", 
            font=('Arial', 24, 'bold'),
            bg='#1a1a1a', 
            fg='#00ff00'
        )
        title_label.pack(pady=10)
        
        # Status indicator
        self.status_label = tk.Label(
            self.root,
            text="Status: Ready",
            font=('Arial', 12),
            bg='#1a1a1a',
            fg='#00ff00'
        )
        self.status_label.pack(pady=5)
        
        # Chat display
        chat_frame = tk.Frame(self.root, bg='#1a1a1a')
        chat_frame.pack(padx=20, pady=10, fill='both', expand=True)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=70,
            height=20,
            bg='#000000',
            fg='#00ff00',
            font=('Consolas', 11),
            insertbackground='#00ff00'
        )
        self.chat_display.pack(fill='both', expand=True)
        
        # Input frame
        input_frame = tk.Frame(self.root, bg='#1a1a1a')
        input_frame.pack(padx=20, pady=10, fill='x')
        
        # Text input
        self.text_input = tk.Entry(
            input_frame,
            font=('Arial', 12),
            bg='#333333',
            fg='#00ff00',
            insertbackground='#00ff00'
        )
        self.text_input.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.text_input.bind('<Return>', self.send_text_message)
        
        # Buttons frame
        buttons_frame = tk.Frame(input_frame, bg='#1a1a1a')
        buttons_frame.pack(side='right')
        
        # Send button
        send_btn = tk.Button(
            buttons_frame,
            text="Send",
            command=self.send_text_message,
            bg='#333333',
            fg='#00ff00',
            font=('Arial', 10, 'bold'),
            relief='raised',
            bd=2
        )
        send_btn.pack(side='left', padx=2)
        
        # Voice input button
        self.voice_btn = tk.Button(
            buttons_frame,
            text="🎤 Listen",
            command=self.toggle_voice_input,
            bg='#333333',
            fg='#00ff00',
            font=('Arial', 10, 'bold'),
            relief='raised',
            bd=2
        )
        self.voice_btn.pack(side='left', padx=2)
        
        # Clear button
        clear_btn = tk.Button(
            buttons_frame,
            text="Clear",
            command=self.clear_chat,
            bg='#333333',
            fg='#00ff00',
            font=('Arial', 10, 'bold'),
            relief='raised',
            bd=2
        )
        clear_btn.pack(side='left', padx=2)
        
        # Initialize chat
        self.add_message("JARVIS", "Hello! I am JARVIS, your AI assistant. How can I help you today?")
    
    def add_message(self, sender, message):
        """Add message to chat display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: {message}\n\n")
        self.chat_display.see(tk.END)
        
        # Store in history
        self.chat_history.append({
            'timestamp': timestamp,
            'sender': sender,
            'message': message
        })
    
    def send_text_message(self, event=None):
        """Send text message"""
        user_input = self.text_input.get().strip()
        if not user_input:
            return
        
        # Clear input
        self.text_input.delete(0, tk.END)
        
        # Add user message
        self.add_message("You", user_input)
        
        # Process in separate thread
        threading.Thread(target=self.process_message, args=(user_input,), daemon=True).start()
    
    def process_message(self, user_input):
        """Process user message and generate response"""
        self.update_status("Processing...")
        
        # Generate response
        response = self.jarvis.generate_response(user_input)
        
        # Update GUI in main thread
        self.root.after(0, self.handle_response, response)
    
    def handle_response(self, response):
        """Handle generated response"""
        # Add JARVIS response
        self.add_message("JARVIS", response)
        
        # Speak response in separate thread
        threading.Thread(target=self.jarvis.speak, args=(response,), daemon=True).start()
        
        self.update_status("Ready")
    
    def toggle_voice_input(self):
        """Toggle voice input"""
        if not self.jarvis.is_listening:
            self.start_voice_input()
        else:
            self.stop_voice_input()
    
    def start_voice_input(self):
        """Start voice input"""
        self.jarvis.is_listening = True
        self.voice_btn.config(text="🛑 Stop", bg='#ff3333')
        self.update_status("Listening...")
        
        # Start listening in separate thread
        threading.Thread(target=self.voice_input_thread, daemon=True).start()
    
    def stop_voice_input(self):
        """Stop voice input"""
        self.jarvis.is_listening = False
        self.voice_btn.config(text="🎤 Listen", bg='#333333')
        self.update_status("Ready")
    
    def voice_input_thread(self):
        """Voice input thread"""
        while self.jarvis.is_listening:
            try:
                # Listen for speech
                speech_text = self.jarvis.listen_for_speech()
                
                if speech_text and self.jarvis.is_listening:
                    # Update GUI in main thread
                    self.root.after(0, self.handle_voice_input, speech_text)
                    break
                    
            except Exception as e:
                self.root.after(0, self.update_status, f"Error: {str(e)}")
                break
        
        # Stop listening
        self.root.after(0, self.stop_voice_input)
    
    def handle_voice_input(self, speech_text):
        """Handle voice input"""
        if speech_text != "Sorry, I couldn't understand that." and \
           speech_text != "Sorry, speech recognition service is unavailable.":
            
            # Add user message
            self.add_message("You (Voice)", speech_text)
            
            # Process message
            threading.Thread(target=self.process_message, args=(speech_text,), daemon=True).start()
        else:
            self.add_message("System", speech_text)
    
    def clear_chat(self):
        """Clear chat display"""
        self.chat_display.delete(1.0, tk.END)
        self.chat_history.clear()
        self.add_message("JARVIS", "Chat cleared. How can I help you?")
    
    def update_status(self, status):
        """Update status label"""
        self.status_label.config(text=f"Status: {status}")
    
    def save_chat_history(self):
        """Save chat history to file"""
        if self.chat_history:
            with open(f'chat_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(self.chat_history, f, indent=2)

def main():
    """Main function"""
    root = tk.Tk()
    app = JARVISGui(root)
    
    # Handle window closing
    def on_closing():
        app.save_chat_history()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()