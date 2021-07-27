from flask import Flask, request
from model import predict_price

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def URLpredict():
    return {
         "result": predict_price(request.args.get(Year,Mileage,State,Make,Model))
        }, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
