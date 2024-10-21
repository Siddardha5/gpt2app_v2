import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Source: https://github.com/CaliberAI/streamlit-nlg-gpt-2/blob/main/app.py
#https://docs.streamlit.io/develop/api-reference/widgets/st.slider

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


# Streamlit app
st.title("GPT-2 Text Generator")

prompt = st.text_input("Enter your prompt:")
num_tokens = st.number_input("Number of tokens to generate:", min_value=1, max_value=100, value=20)

def generate_text(prompt, num_tokens, temperature):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
# Source: https://github.com/huggingface/transformers/issues/22405
   
    outputs = model.generate(
        inputs, 
        max_length=num_tokens + len(inputs[0]),
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if st.button("Generate"):
    if prompt:
        # Generate more predictable text
        predictable_text = generate_text(prompt, num_tokens, temperature=0.2)
        st.subheader("More Predictable Output:")
        st.write(predictable_text)

        # Generate more creative text
        creative_text = generate_text(prompt, num_tokens, temperature=creativity)
        st.subheader(f"More Creative Output (Temperature: {creativity}):")
        st.write(creative_text)
    else:
        st.warning("Please enter a prompt.")
