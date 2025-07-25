import pickle 
from flask import Flask
from flask import request #para archivos json
from flask import jsonify

model_file = 'model_C=1.0.bin' #creamos el nombre del archivo

app = Flask('predict') #esto crea una app de flask cuyo nombre es ping

with open(model_file, 'rb') as f_in: #read-binary = rb
    (dv, model) = pickle.load(f_in) #se cierra automáticamente 

 
@app.route('/predict', methods=['POST']) #declarator, post method sirve para enviar información a la web
def predict():                         #en este caso mandamos un cliente
    customer = request.get_json() #retorna el cuerpo del json como un diccionario

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)

    }

    return jsonify(result) #convierte en json de nuevo el resultado

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
