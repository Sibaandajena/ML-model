from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("C:\\Users\\HP\\Desktop\\house price predication\\house_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        features = [
            float(request.form.get("bedrooms")),
            float(request.form.get("bathrooms")),
            float(request.form.get("sqft_living")),
            float(request.form.get("sqft_lot")),
            float(request.form.get("floors")),
            float(request.form.get("waterfront")),
            float(request.form.get("view")),
            float(request.form.get("condition")),
            float(request.form.get("grade")),
            float(request.form.get("sqft_above")),
            float(request.form.get("sqft_basement")),
            float(request.form.get("yr_built")),
            float(request.form.get("yr_renovated")),
            float(request.form.get("zipcode")),
            float(request.form.get("lat")),
            float(request.form.get("long")),
            float(request.form.get("sqft_living15")),
            float(request.form.get("sqft_lot15"))
        ]

        prediction = model.predict([features])[0]

        return render_template(
            "index.html",
            prediction_text=f"🏡 Predicted House Price: ${prediction:,.2f}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)