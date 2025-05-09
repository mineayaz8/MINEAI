import torch
import numpy as np
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import requests
from pygame import mixer
import time
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import json

class EmotionAIChatbot:
    def __init__(self):
        # Initialize components
        self.load_models()
        self.emotional_state = {
            'joy': 0.5, 'sadness': 0.0, 'anger': 0.0, 
            'fear': 0.0, 'surprise': 0.0, 'neutral': 0.5
        }
        self.memory = []
        self.animation_frames = []
        
    def load_models(self):
        """Load all required ML models"""
        print("Loading AI models...")
        
        # Conversation model
        self.chat_model = pipeline(
            "text-generation", 
            model="microsoft/DialoGPT-large",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Emotion detection
        self.emotion_model = pipeline(
            "text-classification",
            model="bhadresh-savani/bert-base-uncased-emotion",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Search model (for factual queries)
        self.search_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.search_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        
        # Voice synthesis (placeholder - would use actual TTS in production)
        mixer.init()
        
    def detect_emotion(self, text):
        """Analyze text for emotional content"""
        result = self.emotion_model(text)[0]
        return result['label'].lower(), result['score']
    
    def update_emotional_state(self, detected_emotion):
        """Adjust the AI's emotional state based on interaction"""
        emotion, intensity = detected_emotion
        
        # Decay existing emotions
        for e in self.emotional_state:
            self.emotional_state[e] *= 0.8
            
        # Update with new emotion
        self.emotional_state[emotion] += intensity * 0.5
        
        # Normalize
        total = sum(self.emotional_state.values())
        for e in self.emotional_state:
            self.emotional_state[e] /= total
    
    def generate_response(self, user_input):
        """Generate context-aware response"""
        # Detect user emotion
        user_emotion = self.detect_emotion(user_input)
        self.update_emotional_state(user_emotion)
        
        # Check if factual question
        if self.is_factual_query(user_input):
            return self.handle_factual_query(user_input)
        
        # Generate conversational response with emotional tone
        dominant_emotion = max(self.emotional_state.items(), key=lambda x: x[1])[0]
        prompt = f"[{dominant_emotion}] {user_input}"
        
        response = self.chat_model(
            prompt,
            max_length=100,
            pad_token_id=self.chat_model.tokenizer.eos_token_id
        )[0]['generated_text']
        
        # Remove emotion tag from response
        response = response.split(']', 1)[-1].strip()
        
        # Update animation frames
        self.generate_animation_frames(dominant_emotion)
        
        return response
    
    def is_factual_query(self, text):
        """Check if the input is a factual question"""
        factual_triggers = ["what is", "who is", "when did", "how many", "define"]
        return any(trigger in text.lower() for trigger in factual_triggers)
    
    def handle_factual_query(self, query):
        """Handle factual questions using search augmentation"""
        # Generate search query
        inputs = self.search_tokenizer(
            f"question: {query} context:", 
            return_tensors="pt",
            max_length=512, 
            truncation=True
        )
        
        # Get search results (simplified - would use actual API in production)
        search_results = self.mock_search(query)
        
        # Generate answer
        outputs = self.search_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=200
        )
        
        answer = self.search_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return f"According to my research: {answer}\n\n[Source: {search_results}]"
    
    def mock_search(self, query):
        """Simulate search results (replace with actual API calls)"""
        mock_data = {
            "what is ai": "AI stands for Artificial Intelligence...",
            "who is elon musk": "Entrepreneur behind Tesla and SpaceX...",
            "capital of france": "Paris is the capital of France..."
        }
        return mock_data.get(query.lower(), "Various online sources")
    
    def generate_animation_frames(self, emotion):
        """Generate simple animation frames based on emotion"""
        self.animation_frames = []
        
        # Create emotion-specific animation
        if emotion == 'joy':
            self.animation_frames = ["(^_^)", "(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧", "✧◝(⁰▿⁰)◜✧"]
        elif emotion == 'sadness':
            self.animation_frames = ["(´･_･`)", "(╥﹏╥)", "(っ˘̩╭╮˘̩)っ"]
        elif emotion == 'anger':
            self.animation_frames = ["(╬ Ò﹏Ó)", "ಠ_ಠ", "(ﾒ` ﾛ ´)︻デ═一"]
        else:  # neutral/default
            self.animation_frames = ["(•_•)", "( •_•)>⌐■-■", "(⌐■_■)"]
    
    def speak(self, text):
        """Simple text-to-speech simulation"""
        print(f"\nAI: {text}")
        for frame in self.animation_frames:
            print(f"\r{frame}", end="", flush=True)
            time.sleep(0.3)
        print("\n")
        
    def visualize_emotions(self):
        """Show current emotional state"""
        plt.figure(figsize=(8, 4))
        plt.bar(self.emotional_state.keys(), self.emotional_state.values())
        plt.title("Current Emotional State")
        plt.ylabel("Intensity")
        plt.ylim(0, 1)
        plt.show()

# Demo execution
if __name__ == "__main__":
    bot = EmotionAIChatbot()
    
    print("Advanced AI Chatbot initialized. Type 'quit' to exit.")
    bot.visualize_emotions()
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        response = bot.generate_response(user_input)
        bot.speak(response)
        bot.visualize_emotions()
