from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model_path = "models/svm_gamma=0.001_C=0.7_.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"

@app.route("/predict", methods=['POST'])
def predict_digit_1():
    image = request.json['image']
    print("done loading")
    predicted = model.predict([image])
    return {"y_predicted":int(predicted[0])}

if __name__ == "__main__":
	app.run(host ="0.0.0.0", port = int("8000"))