from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("Dataset.csv")
app = Flask(__name__)

model = pickle.load(open('model1.pkl', 'rb'))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        features = [int(x) for x in request.form.values()]
        print(features)
        final = [np.array(features)]
        x = df.iloc[:, 0:24].values
        sst = StandardScaler()
        sst.fit(x)
        output = model.predict(sst.transform(final))
        print(output)

        if output[0] == 0:
            return render_template('predict.html', prediction=0)
        else:
            return render_template('predict.html', prediction=1)


if __name__ == '__main__':
    app.run(debug=True)
