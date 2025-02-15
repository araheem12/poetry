import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained model and tokenizer
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dukhi_shayari.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()
max_sequence_length = model.input_shape[1]
total_words = model.output_shape[1]

def generate_poetry(seed_text, next_words=20, temperature=0.5):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predictions = model.predict(token_list, verbose=0)[0]
        predictions = np.log(predictions + 1e-8) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        predicted_index = np.random.choice(len(predictions), p=predictions)
        output_word = next((word for word, index in tokenizer.word_index.items() if index == predicted_index), "")
        seed_text += " " + output_word
    return seed_text

# Streamlit UI
st.title("Roman Urdu Poetry Generator")
st.write("Generate poetry by providing a seed text")

seed_text = st.text_input("Enter a seed text:", "sunāo")
num_words = st.slider("Number of words to generate:", 5, 50, 20)
temperature = st.slider("Creativity Level (Temperature):", 0.1, 1.5, 0.7, 0.1)

if st.button("Generate Poetry"):
    generated_poetry = generate_poetry(seed_text, num_words, temperature)
    st.subheader("Generated Poetry")
    st.write(generated_poetry)
