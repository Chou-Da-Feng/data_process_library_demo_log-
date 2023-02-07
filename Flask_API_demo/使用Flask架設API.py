from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS:跨來源資源共享
import numpy as np
import func.model as model

app = Flask(__name__)
CORS(app)

m = model.xgboostModel


@app.route('/')
def index():
    return "hello!!"


@app.route('/predict', methods=['POST'])  # 預設為GET
def postInput():
    # 取得前端傳過來的數值
    inserValues = request.get_json()
    x1 = inserValues['sepal.length']
    x2 = inserValues['sepal.width']
    x3 = inserValues['petal.length']
    x4 = inserValues['petal.width']
    input = np.array([[x1, x2, x3, x4]])

    # m = model.xgboostModel
    # result = m.predict(input)
    #    print(input)

    return jsonify({'return': str(m.predict(input)[0])})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)

