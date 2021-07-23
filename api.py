from flask import Flask, request
from m09_model_deployment import predict_proba

app = Flask(__name__)

@app.route('/CarPricing', methods=['GET'])
def CarPricingPredict():
    return {
         "result": Heidy_FabiModel(request.args.get('URL'))
        }, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
