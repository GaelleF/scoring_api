from flask import Flask, request

app = Flask(__name__)

# Config options - Make sure you created a 'config.py' file.
app.config.from_object('config')
# To get one variable, tape app.config['MY_VARIABLE']

from .utils_ml import model_predict


@app.route('/')
def index():
    return "Hello world !"


@app.route('/predict/', methods=['POST'])
def predict():
  #getting an array of features from the post request's body
  payload = request.form.to_dict()
  #feature_array = np.fromstring(query_parameters['feature_array'],dtype=float,sep=",")
  print('QUERRRY', payload)
  response = model_predict(payload)
  print('RESPONSE', response)
  return response