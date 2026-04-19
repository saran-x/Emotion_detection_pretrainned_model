"""
predict_with_mapping.py - Load labels from saved model
"""

from transformers import pipeline
import json
import os

model_path = "../model/emotion_model_final"

# Load label mapping
try:
    with open(f"{model_path}/label_mapping.json", 'r') as f:
        mapping = json.load(f)
        if 'id_to_label' in mapping:
            # Convert string keys to int, then to LABEL_X format
            label_to_emotion = {}
            for idx, emotion in mapping['id_to_label'].items():
                label_to_emotion[f'LABEL_{idx}'] = emotion
        else:
            label_to_emotion = {
                'LABEL_0': 'sadness', 'LABEL_1': 'joy', 'LABEL_2': 'love',
                'LABEL_3': 'anger', 'LABEL_4': 'fear', 'LABEL_5': 'surprise'
            }
except:
    # Fallback mapping
    label_to_emotion = {
        'LABEL_0': 'sadness', 'LABEL_1': 'joy', 'LABEL_2': 'love',
        'LABEL_3': 'anger', 'LABEL_4': 'fear', 'LABEL_5': 'surprise'
    }

print(f"📊 Label mapping: {label_to_emotion}")

# Load model
classifier = pipeline("text-classification", model=model_path)

print("\n" + "="*50)
print("EMOTION RECOGNITION")
print("="*50)

while True:
    text = input("\n📝 Enter text (or 'quit'): ")
    
    if text.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    result = classifier(text)
    label = result[0]['label']
    confidence = result[0]['score']
    emotion = label_to_emotion.get(label, label)
    
    # Emoji mapping
    emojis = {
        'sadness': '😢', 'joy': '😊', 'love': '❤️',
        'anger': '😠', 'fear': '😨', 'surprise': '😲'
    }
    emoji = emojis.get(emotion, '📊')
    
    print(f"\n   Text: {text}")
    print(f"   → {emoji} {emotion.upper()}: {confidence:.2%}")
    print("-" * 40)