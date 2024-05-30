from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

# Load and preprocess data
job_data = pd.read_csv('job_listings.csv')
user_data = pd.read_csv('user_profiles.csv')
nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

job_data['processed_description'] = job_data['description'].apply(preprocess)
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(job_data['processed_description'])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
user_item_matrix = np.random.rand(user_data.shape[0], job_data.shape[0])

def hybrid_recommendations(user_id):
    content_scores = similarity_matrix.dot(user_item_matrix[user_id])
    collaborative_scores = user_item_matrix[user_id]
    combined_scores = 0.5 * content_scores + 0.5 * collaborative_scores
    return np.argsort(combined_scores)[::-1][:5]

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        try:
            user_name = user_data.loc[user_data['user_id'] == user_id, 'name'].values[0]
        except IndexError:
            return "User ID not found", 404
        recommended_jobs = hybrid_recommendations(user_id)
        recommended_jobs_data = job_data.iloc[recommended_jobs]
        return render_template('index.html', user={'name': user_name}, matched_jobs=recommended_jobs_data.to_dict('records'))
    
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
