
from flask import Flask, request, jsonify
import time
import threading
import random
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

# Initialize LLM (Sentence Transformer)
model = SentenceTransformer('all-MiniLM-L6-v2')  

# Initialize FAISS
dimension = 384  # Dimension of the embeddings produced by 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(dimension)  # Use L2 distance for similarity search
responses = []  # Store the text of responses for retrieval after FAISS search

# Simulated emergency response database
emergency_responses = {
    "not breathing": "perform CPR by pushing firmly downwards in the middle of the chest and then releasing.",
    "bleeding": "apply pressure to the wound with a clean cloth.",
    "choking": "perform the Heimlich maneuver by standing behind the person and using your hands to exert pressure on the bottom of the diaphragm."
}

# Populate the FAISS index with emergency response embeddings
for key, response in emergency_responses.items():
    embedding = model.encode(key).astype('float32')  # FAISS requires float32 type
    index.add(np.array([embedding]))  # Add to FAISS index
    responses.append(response)  # Keep track of the response text

def get_emergency_response(query):
    """Retrieve the most relevant emergency response using FAISS."""
    query_embedding = model.encode(query).astype('float32')  # Encode and convert to float32
    distances, indices = index.search(np.array([query_embedding]), k=1)  # Find the nearest neighbor
    if len(indices) > 0 and len(indices[0]) > 0 and indices[0][0] < len(responses):
        return responses[indices[0][0]]  # Return the most relevant response
    return "Call 911 immediately."

@app.route('/')
def welcome():
    return '''
    <html><head>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; }
        h1 { color: #333; }
        form { margin-top: 20px; }
        input[type="text"], textarea { padding: 10px; width: 300px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ccc; }
        input[type="submit"] { padding: 10px 20px; background-color: #5cb85c; color: white; border: none; border-radius: 5px; cursor: pointer; }
        input[type="submit"]:hover { background-color: #4cae4c; }
    </style>
    </head><body>
    <h1>Hello, are you having an emergency or would you like to leave a message? (emergency/message)</h1>
    <form action="/handle_input" method="post">
        <input type="text" name="text" placeholder="Enter 'emergency' or 'message'"/>
        <input type="submit" value="Submit"/>
    </form>
    </body></html>
    '''

@app.route('/handle_input', methods=['POST'])
def handle_input():
    user_input = request.form.get('text', '').lower()
    
    if user_input == 'emergency':
        return '''
        <html><body>
        What is the emergency? 
        <form action="/emergency_response" method="post">
            <input type="text" name="emergency_type" placeholder="Enter the type of emergency"/>
            <input type="submit" value="Submit"/>
        </form>
        </body></html>
        '''
    elif user_input == 'message':
        return '''
        <html><body>
        Please leave your message.
        <form action="/message" method="post">
            <textarea name="text"></textarea>
            <input type="submit" value="Submit"/>
        </form>
        </body></html>
        '''
    else:
        return "I don't understand that. Are you having an emergency or would you like to leave a message? (emergency/message)"

@app.route('/emergency_response', methods=['POST'])
def emergency_response():
    emergency_type = request.form.get('emergency_type', '').lower()
    response = get_emergency_response(emergency_type)
    return f"Emergency instructions: {response}. Meanwhile, can you tell me which area are you located right now?<br><form action='/location' method='post'><input type='text' name='text' placeholder='Enter your location'/><input type='submit' value='Submit'/></form>"

@app.route('/message', methods=['POST'])
def message():
    user_message = request.form.get('text', '')
    return f"Thanks for the message, we will forward it to Dr. Adrin: {user_message}"

@app.route('/location', methods=['POST'])
def location():
    user_location = request.form.get('text', '')
    eta = random.randint(10, 30)  # Random ETA between 10 and 30 minutes
    return f"Dr. Adrin will be at your location in approximately {eta} minutes. Please follow the emergency instructions provided. If the situation worsens, call 911 immediately."

if __name__ == '__main__':
    app.run(debug=True)

