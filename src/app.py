from flask import Flask
from flask import jsonify
from flask import make_response
from flask import request


def lil_func(x):
    return x + 1


app = Flask(__name__)


@app.route("/", methods=['GET'])
def health_check():
    return "Alive!"


@app.route("/predict/<uuid:user>")
def predict(user):
    return make_response(jsonify("not implemented"), 500)


@app.route("/train/<uuid:user>/<uuid:poi>")
def train(user, poi):
    return make_response(jsonify({"user": user, "poi": poi, 'message': "Not implemented :P"}), 500)


if __name__ == '__main__':
    app.run(debug=True)
