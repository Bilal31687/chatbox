# Import necessary libraries
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Load the 'flan-t5-base' model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a chatbot pipeline using text2text generation
chatbot = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Define the Streamlit app UI and interaction logic
def main():
    st.title("🤖 AI Chatbot: Learn About Artificial Intelligence")
    st.write("Ask me anything about AI, machine learning, or related topics!")

    # Add animation for fun!
    st.image("https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif", width=300)

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Input area for user questions
    user_input = st.text_input("You: ", "")

    if user_input:
        with st.spinner("Thinking..."):
            # Generate response from the model
            response = chatbot(user_input, max_length=100, do_sample=True)[0]['generated_text']
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Chatbot", response))

    # Display the chat history
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**{sender}:** {message}")
        else:
            st.markdown(f"**{sender}**: {message}")

if __name__ == "__main__":
    main()
