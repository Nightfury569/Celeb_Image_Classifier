from flask import Flask, request, jsonify
import util

app = Flask(__name__)
app.secret_key='Super-secret2'

@app.route('/img_classify_log_reg',methods=['GET', 'POST'])     # Prediction using logistic_regression
def imageClassification():
    image_data = request.form['image_data']

    response = jsonify(util.classify_image(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/img_classify_svm',methods=['GET', 'POST'])      #Prediction using svm
def imageClassificationSvm():
    image_data = request.form['image_data']

    response = jsonify(util.classify_image_svm(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route('/json_data', methods=['GET'])
def class_dictionary_data():
    return jsonify(util.__name_to_number)

if __name__ == "__main__":
    util.load_artifacts()
    app.run(debug=True)