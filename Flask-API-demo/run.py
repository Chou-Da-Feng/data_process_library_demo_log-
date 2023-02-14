# -*- coding: UTF-8 -*-
import numpy as np
import model

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return 'hello!!'


@app.route('/predict', methods=['POST'])
def postInput():
    # 取得前端傳過來的數值
    insertValues = request.get_json()
    x1 = insertValues['sepal.length']
    x2 = insertValues['sepal.width']
    x3 = insertValues['petal.length']
    x4 = insertValues['petal.width']
    input = np.array([[x1, x2, x3, x4]])

    result = model.predict(input)

    return jsonify({'return': str(result)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
