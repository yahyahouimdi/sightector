from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer



app = Flask(__name__)

with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

#Load the fitted vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=pickle.load(open("tfidfvectoizer.pkl", "rb")))
model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))
# Load the saved pipeline (it includes both TfidfVectorizer and LinearSVC)
#with open("harassment_model_pipeline.pkl", "rb") as model_file:
    #model_pipeline = pickle.load(model_file)
# Existing route for rendering the form-based interface
@app.route('/SecureHerServer')
def dashboard():
    return render_template('server.html')

# New route for handling JSON POST requests from the extension
@app.route('/', methods=['GET', 'POST'])
def detect():
    try:
        data = request.get_json()  # Get the JSON data from the request
        user_input = data.get('text', '')  # Extract the 'text' field from the JSON
        
        if not user_input:
            return jsonify({"error": "No text provided"}), 400
        
        # Transform the input using the pre-loaded vectorizer
        # transformed_input = vectorizer.transform([user_input])
       
        transformed_input = vectorizer.fit_transform([user_input])             
        
        # Make the prediction using the model
        #prediction = model_pipeline.predict([user_input])[0]
        prediction = model.predict(transformed_input)[0]
        # Return the result as a JSON response
        return jsonify({"harassment": bool(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




    

if __name__ == '__main__':
    app.run(debug=True)