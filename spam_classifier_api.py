import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM, Embedding
from tensorflow.keras.models import Model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load the dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.columns = ['labels', 'data']
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].values
df_train, df_test, Ytrain, Ytest = train_test_split(df['data'], Y, test_size=0.33)

# Tokenization
MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df_train)
sequences_train = tokenizer.texts_to_sequences(df_train)
sequences_test = tokenizer.texts_to_sequences(df_test)
word2idx = tokenizer.word_index
V = len(word2idx)
T = max(len(s) for s in sequences_train)
data_train = pad_sequences(sequences_train, maxlen=T)
data_test = pad_sequences(sequences_test, maxlen=T)

# Model definition
D = 20
M = 15
i = Input(shape=(T,))
x = Embedding(V + 1, D)(i)
x = LSTM(M, return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(i, x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
print('Training model...')
r = model.fit(data_train, Ytrain, epochs=10, validation_data=(data_test, Ytest))

# FastAPI app definition
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Allow these HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define input model for API endpoint
class InputData(BaseModel):
    message: str

# Define API endpoint for prediction
@app.post("/predict")
async def predict_spam(input_data: InputData):
    # Process input data
    my_test_case = [input_data.message]
    sequences_my_test = tokenizer.texts_to_sequences(my_test_case)
    data_my_test = pad_sequences(sequences_my_test, maxlen=T)

    # Make predictions
    predictions = model.predict(data_my_test)
    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int)
    predicted_messages = ["spam" if label == 1 else "no spam" for label in predicted_labels]

    # Prepare response
    response = {
        "raw_predictions": predictions.tolist(),
        "predicted_labels": predicted_labels.tolist(),
        "predicted_messages": predicted_messages
    }
    return response

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
