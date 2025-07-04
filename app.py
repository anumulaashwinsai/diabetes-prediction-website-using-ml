from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
print("Loading model...")
try:
    model = joblib.load('diabetes_model_optimal.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Features: {len(feature_names)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    feature_names = None

@app.route('/')
def home():
    try:
        with open('index.html', 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"<h1>Diabetes Prediction App</h1><p>Error: {e}</p>"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        print(f"Received data: {data}")
        
        # Create feature array in correct order
        features = []
        missing_features = []
        
        for feature_name in feature_names:
            if feature_name in data:
                features.append(float(data[feature_name]))
            else:
                missing_features.append(feature_name)
        
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        # Get probabilities
        try:
            probabilities = model.predict_proba(features_array)[0]
            prob_dict = {
                'no_diabetes': float(probabilities[0]),
                'prediabetes': float(probabilities[1]) if len(probabilities) > 2 else 0.0,
                'diabetes': float(probabilities[-1])
            }
        except:
            prob_dict = None
        
        result = {
            'prediction': int(prediction),
            'probabilities': prob_dict
        }
        
        print(f"Prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

    
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

