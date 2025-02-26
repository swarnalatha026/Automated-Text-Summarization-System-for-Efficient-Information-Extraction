import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Sample text data
texts = [
    "Natural Language Processing is a field of AI that focuses on the interaction between computers and humans.",
    "Deep learning models like LSTMs and Transformers are widely used for text summarization.",
    "Text summarization helps in reducing the length of a text while preserving important information."
]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post')

# Define model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=50, input_length=padded_sequences.shape[1]),
    LSTM(100, return_sequences=True),
    LSTM(50),
    Dense(20, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Dummy training labels
y_train = np.array([1, 1, 1])  # Dummy labels

# Train model
model.fit(padded_sequences, y_train, epochs=5)

print("Text summarization model training complete!")