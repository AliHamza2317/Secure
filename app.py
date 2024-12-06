import pandas as pd
import pickle
import os
import hashlib
import json
import re
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, request, jsonify
from flask_cors import CORS
# from pyngrok import ngrok
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoModel, AutoTokenizer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

print(torch.__version__)

app = Flask(__name__)
CORS(app) 

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

original_labels = ['usability', 'security', 'both', 'other issue']
label_encoder = LabelEncoder()
label_encoder.fit(original_labels)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
    
# Load the label encoder from the pickle file
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the models and tokenizers
sentiment_model = AutoModel.from_pretrained('AliHamza2317/sentiment-model').to(device)
sentiment_tokenizer = AutoTokenizer.from_pretrained('AliHamza2317/sentiment-model')
issue_model = AutoModel.from_pretrained('AliHamza2317/issue-model').to(device)
issue_tokenizer = AutoTokenizer.from_pretrained('AliHamza2317/issue-model')
subcategory_model = AutoModel.from_pretrained('AliHamza2317/subcategory-model').to(device)
subcategory_tokenizer = AutoTokenizer.from_pretrained('AliHamza2317/subcategory-model')



pattern_mapping = {
    'Learnability': ['Informative Dialogues', 'User-Friendly Error Messages', 'Conveying Threats & Consequences', 'Human-Computer Interaction for Security (HCI-S)'],
    'Memorability': ['Informative Dialogues', 'User-Friendly Error Messages', 'Conveying Threats & Consequences', 'Human-Computer Interaction for Security (HCI-S)'],
    'User Error Protection': ['Warn When Unsafe', 'Recoverable Errors', 'Disclose Significant Deviations', 'Security Indicators'],
    'Operability': ['Direct Access to UI Components', 'Minimize User Inconvenience', 'Progressive Disclosure', 'Interactive Feedback'],
    'Accessibility': ['Direct Access to UI Components', 'User-Friendly Security Policies', 'Contextual Help and Support'],
    'Satisfaction': ['Attractive Options', 'Interactive Feedback', 'User Consent for Actions', 'Visual and Auditory Notifications'],
    'Efficiency': ['Progressive Disclosure', 'Minimize User Inconvenience', 'System Defaults to Secure Settings'],
    'Confidentiality': ['Encryption and Decryption Options', 'Access Control Management', 'Secure Development Practices', 'Security Testing and Evaluation'],
    'Integrity': ['Encryption and Decryption Options', 'Authorization', 'Incident Response Plans'],
    'Availability': ['Regular Security Updates', 'System Defaults to Secure Settings', 'General Notifications About Security'],
    'Authenticity': ['Email-Based Identification and Authentication', 'Multi-Factor Authentication (MFA)', 'Create a Security Lexicon'],
    'Accountability': ['Audit and Log User Actions', 'Risk Assessment Tools', 'Migrate and Backup Keys'],
    'Non-repudiation': ['Audit and Log User Actions', 'Disclose Significant Deviations', 'Create a Security Lexicon'],
    'Traceability': ['Disclose Significant Deviations', 'Audit and Log User Actions', 'Security Features Used by the User'],
    'Authorization': ['Access Control Management', 'Authorization', 'Security Features Used by the System'],
    'Resilience': ['System Defaults to Secure Settings', 'Regular Security Updates', 'Incident Response Plans']
}

# Function to get patterns based on subcategory
def get_patterns(subcategory):
    return pattern_mapping.get(subcategory, [])


# Define subcategories
usability_columns = ['Learnability', 'Memorability', 'User Error Protection', 'Operability', 'Accessibility', 'Satisfaction', 'Efficiency']
security_columns = ['Confidentiality', 'Integrity', 'Availability', 'Authenticity', 'Accountability', 'Non-repudiation', 'Traceability', 'Authorization', 'Resilience']

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    else:
        text = ''
    return text

def generate_cache_key(data_dict):
    data_string = json.dumps(data_dict, sort_keys=True)  # Convert data to JSON string, ensuring consistent key order
    return hashlib.md5(data_string.encode('utf-8')).hexdigest()


def get_subcategory(predictions, issue_type):
    predictions = np.array(predictions)
    if issue_type == 'usability':
        max_idx = np.argmax(predictions[:len(usability_columns)])
        return usability_columns[max_idx] if predictions[:len(usability_columns)].size > 0 else 'None'
    elif issue_type == 'security':
        max_idx = np.argmax(predictions[:len(security_columns):])
        return security_columns[max_idx] if predictions[:len(security_columns):].size > 0 else 'None'
    elif issue_type == 'both':
        usability_preds = predictions[:len(usability_columns)]
        security_preds = predictions[:len(security_columns):]
        usability_subcategory = usability_columns[np.argmax(usability_preds)] if usability_preds.size > 0 else 'None'
        security_subcategory = security_columns[np.argmax(security_preds)] if security_preds.size > 0 else 'None'
        return f"{usability_subcategory}, {security_subcategory}"
    return 'None'



@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if a file is included in the request
        if 'file' not in request.files:
            print("No file part in request")
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return jsonify({"error": "No selected file"}), 400

        # Load the dataset
        global data
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            data = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format"}), 400
#         data = data.head(00)
        data['Apps Reviews'] = data['Apps Reviews'].astype(str).apply(preprocess_text)

        # Generate cache key
        cache_key = generate_cache_key(data.to_dict())
        cache_file = f'cache/{cache_key}.json'

        # Check if cached file exists
        if os.path.exists(cache_file):
            print(f"Cache hit: {cache_file}")
            with open(cache_file, 'r') as f:
                result = json.load(f)
            return jsonify(result)

        # Define a static user_id for recommendations
        user_id = 1  # or any appropriate user_id

        # Processing functions
        def sentiment_analysis(reviews):
            inputs = sentiment_tokenizer(reviews, padding=True, truncation=True, max_length=128, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            batch_size = 16
            all_preds = []

            sentiment_model.eval()
            with torch.no_grad():
                for i in range(0, input_ids.size(0), batch_size):
                    end = min(i + batch_size, input_ids.size(0))
                    batch_input_ids = input_ids[i:end]
                    batch_attention_mask = attention_mask[i:end]

                    outputs = sentiment_model(batch_input_ids, attention_mask=batch_attention_mask)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_preds.extend(preds)

            return all_preds

        def issue_identification(reviews):
            inputs = issue_tokenizer(reviews, padding=True, truncation=True, max_length=128, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            batch_size = 16
            all_preds = []

            issue_model.eval()
            with torch.no_grad():
                for i in range(0, input_ids.size(0), batch_size):
                    end = min(i + batch_size, input_ids.size(0))
                    batch_input_ids = input_ids[i:end]
                    batch_attention_mask = attention_mask[i:end]

                    outputs = issue_model(batch_input_ids, attention_mask=batch_attention_mask)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_preds.extend(preds)

            return all_preds

        def subcategory_classification(reviews):
            inputs = subcategory_tokenizer(reviews, padding=True, truncation=True, max_length=128, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            batch_size = 16
            all_preds = []

            subcategory_model.eval()
            with torch.no_grad():
                for i in range(0, input_ids.size(0), batch_size):
                    end = min(i + batch_size, input_ids.size(0))
                    batch_input_ids = input_ids[i:end]
                    batch_attention_mask = attention_mask[i:end]

                    outputs = subcategory_model(batch_input_ids, attention_mask=batch_attention_mask)
                    logits = outputs.logits
                    all_preds.extend(logits.cpu().numpy())

            return all_preds

        # Process the dataset in batches
        batch_size = 500  # Changed to avoid possible indexing issues with large batches
        results = []

        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch = data.iloc[start:end].copy()

            reviews = batch['Apps Reviews'].tolist()
            print(f"Processing batch: {start} to {end}")

            try:
                sentiment_preds = sentiment_analysis(reviews)
                issue_preds = issue_identification(reviews)
                subcategory_preds = subcategory_classification(reviews)

                sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
                batch['Sentiment_Result'] = [sentiment_mapping[pred] for pred in sentiment_preds]
                batch['Identify issue'] = label_encoder.inverse_transform(issue_preds)
                batch['Subcategory'] = [get_subcategory(pred, issue) for pred, issue in zip(subcategory_preds, batch['Identify issue'])]

                batch.loc[batch['Identify issue'] == 'other issue', 'Subcategory'] = 'None'

                # Create the 'Recommendation' column based on 'Subcategory'
                batch['Recommendation'] = batch['Subcategory'].apply(lambda subcategory: get_patterns(subcategory) if subcategory != 'None' else 'None')

                # TF-IDF Matrix Calculation
                tfidf_reviews = TfidfVectorizer(stop_words='english')
                tfidf_subcategory = TfidfVectorizer(stop_words='english')

                tfidf_matrix_reviews = tfidf_reviews.fit_transform(batch['Apps Reviews'].fillna(''))
                tfidf_matrix_subcategory = tfidf_subcategory.fit_transform(batch['Subcategory'].fillna(''))

                cosine_sim_reviews = linear_kernel(tfidf_matrix_reviews, tfidf_matrix_reviews)
                cosine_sim_subcategory = linear_kernel(tfidf_matrix_subcategory, tfidf_matrix_subcategory)

#                 Content-Based Recommendation Functions
                def get_content_based_recommendations_reviews(index, cosine_sim=cosine_sim_reviews):
                    if index >= len(batch):
                        print(f"Index {index} out of bounds for batch size {len(batch)}")
                        return []
                    sim_scores = list(enumerate(cosine_sim[index]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    sim_scores = sim_scores[1:6]  # Get the top 5 most similar items
                    item_indices = [i[0] for i in sim_scores]
                    recommendations = [item for sublist in batch['Recommendation'].iloc[item_indices].tolist() for item in sublist if item != 'No Pattern']
                    return recommendations

                def get_content_based_recommendations_subcategory(index):
                    if index >= len(batch):
                        print(f"Index {index} out of bounds for batch size {len(batch)}")
                        return []
                    try:
                        sim_scores = list(enumerate(cosine_sim_subcategory[index]))
                        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                        sim_scores = sim_scores[1:6]  # Get the top 5 most similar items
                        item_indices = [i[0] for i in sim_scores]
                        recommendations = [item for sublist in batch['Recommendation'].iloc[item_indices].tolist() for item in sublist if item != 'No Pattern']
                        return recommendations
                    except Exception as e:
                        print(f"Error in get_content_based_recommendations_subcategory: {e}")
                        return []

                # Collaborative Filtering
                interaction_data = {
                    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                    'item_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'rating': [5, 3, 4, 2, 5, 1, 3, 4, 2, 1]
                }
                interaction_df = pd.DataFrame(interaction_data)
                reader = Reader(rating_scale=(1, 5))
                interaction_dataset = Dataset.load_from_df(interaction_df[['user_id', 'item_id', 'rating']], reader)
                trainset, testset = train_test_split(interaction_dataset, test_size=0.25)

                svd = SVD()
                svd.fit(trainset)
                
                def get_collaborative_recommendations(user_id, svd, n=5):
                    try:
                        if user_id not in interaction_df['user_id'].values:
                            print(f"User ID {user_id} not found in the interaction data.")
                            return []
                        item_ids = interaction_df['item_id'].unique()
                        predictions = [svd.predict(user_id, item_id) for item_id in item_ids]
                        predictions.sort(key=lambda x: x.est, reverse=True)
                        recommended_items = [pred.iid for pred in predictions[:n]]
                        recommendations = [item for sublist in batch['Recommendation'].iloc[recommended_items].tolist() for item in sublist if item != 'No Pattern']
                        return recommendations
                    except Exception as e:
                        print(f"Error in get_collaborative_recommendations: {e}")
                        return []

                def hybrid_recommendations(user_id, index, n=5):
                    
                    content_recs_reviews = get_content_based_recommendations_reviews(index)
                    content_recs_subcategory = get_content_based_recommendations_subcategory(index)
                    collab_recs = get_collaborative_recommendations(user_id, svd, n)
                    hybrid_recs = list(set(content_recs_reviews + content_recs_subcategory + collab_recs))
                    return [rec for rec in hybrid_recs if rec]

                # Generate hybrid recommendations for the batch
                batch['Hybrid_Recommendations'] = [
                    hybrid_recommendations(user_id, i) if batch['Subcategory'].iloc[i] != 'None' else 'None'
                    for i in range(len(batch))
                ]

                negative_reviews = batch[batch['Sentiment_Result'] == 'negative']
                results.extend(negative_reviews[['Apps Reviews', 'Sentiment_Result', 'Identify issue', 'Subcategory', 'Hybrid_Recommendations']].to_dict(orient='records'))

            except Exception as batch_e:
                print(f"Error processing batch {start} to {end}: {batch_e}")

        # Cache the result
        if not os.path.exists('cache'):
            os.makedirs('cache')
        with open(cache_file, 'w') as f:
            json.dump(results, f)

        print(f"Total results: {results}")
        return jsonify(results)

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Set up Flask to run with ngrok
    # port = 5016

    # print(f'Flask app running at {public_url}')
 
    app.run(host='0.0.0.0', port=5016)



