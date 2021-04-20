import pickle
from flask import Flask, request, jsonify
from model_files.nlp_model import predict



app = Flask("qc_prediction")

@app.route('/', methods=['POST'])
def predict_qc():
    json_data = request.get_json()

    # if json_data is None or 'text' not in json:
    #         return 'Request does not contain valid JSON with the question attribute!', 400

    # question = json_data['text']

    
    # with open('./model_files/model_coarse.bin', 'rb') as f_in:
    #     model_coarse = pickle.load(f_in)
    #     f_in.close()
    
    # with open('./model_files/model_coarse.bin', 'rb') as f_in:
    #     model_fine = pickle.load(f_in)
    #     f_in.close()
    
    # with open('./model_files/le_coarse', 'rb') as f_in:
    #     le_coarse = pickle.load(f_in)
    #     f_in.close()
    
    # with open('./model_files/le_fine', 'rb') as f_in:
    #     le_fine = pickle.load(f_in)
    #     f_in.close()
    
    # with open('./model_files/count_vecs', 'rb') as f_in:
    #     count_vecs = pickle.load(f_in)
    #     f_in.close()
    
    predictions = predict(json_data)

    response = {
        'coarse_class': predictions[0],
        'fine_class': predictions[1]
    }

    return jsonify(response)



    



@app.route('/', methods=['GET'])
def ping():
    return "Pinging model!"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)