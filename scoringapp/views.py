from flask import Flask

app = Flask(__name__)

# Config options - Make sure you created a 'config.py' file.
app.config.from_object('config')
# To get one variable, tape app.config['MY_VARIABLE']

from .utils_ml import model_predict


@app.route('/')
def index():
    return "Hello world !"

#, methods=['POST']
@app.route('/predict/')
def predict():
  #getting an array of features from the post request's body
  #query_parameters = request.args
  #feature_array = np.fromstring(query_parameters['feature_array'],dtype=float,sep=",")
  response = model_predict()
  print('RESPONSE', response)
  return model_predict()