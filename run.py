import joblib
import os

# Load the trained model
model_path = "model.pkl"
if not os.path.exists(model_path):
    print("❌ model.pkl not found. Run model.py first.")
    exit(1)

model = joblib.load(model_path)
print("✅ Model loaded successfully!\n")

print("=== AI Spam Detector ===")
print("Type 'exit' to quit.\n")

while True:
    message = input("Enter your message: ")
    if message.lower() == "exit":
        break
    result = model.predict([message])[0]
    prediction = "🚨 SPAM" if result == 1 else "✅ NOT SPAM"
    print(f"Prediction: {prediction}\n")
