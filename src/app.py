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


@app.route("/predict/<uuid:user>", methods=['POST'])
def predict(user):
    assert request.is_json
    content = request.json
    return make_response(jsonify({'user': user, 'location': content['location']}), 500)


@app.route("/train/<uuid:user>/<uuid:poi>")
def train(user, poi):
    return make_response(jsonify({"user": user, "poi": poi, 'message': "Not implemented :P"}), 500)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
