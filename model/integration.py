from flask import Flask, request, jsonify
from model import ProblemModel  # Importing our model class from model.py
import torch

app = Flask(__name__)

# Loading our model instance
mi = ProblemModel()  # Initializing our model

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_prompt = data['prompt']

        # Using our model's get_solution method
        solution = mi.get_solution(user_prompt)

        return jsonify({'solution': solution})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':

    app.run(debug=True)
