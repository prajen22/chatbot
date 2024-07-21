import os
import streamlit as st
from groq import Groq
from textblob import TextBlob
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

# Set the environment variable for the API key
os.environ['GROQ_API_KEY'] = "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"

# Load pre-trained model and image processor for image classification
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Function to analyze the image
def analyze_image(image):
    # Ensure the image is in RGB format
    image = image.convert("RGB")
    
    # Preprocess the image
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # Get the predicted label
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    return f"The image is classified as: {predicted_class}"

# Initialize Groq client
def get_chat_completion(messages):
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        return "API key is not set. Please check the environment variable."
    
    client = Groq(api_key=api_key)
    
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def main():
    st.title("Multimodel Chatbot Service")

    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'feedback_file' not in st.session_state:
        st.session_state.feedback_file = "feedback.txt"
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = ""

    # User authentication (dummy example)
    if not st.session_state.authenticated:
        username = st.text_input("Username:")
        password = st.text_input("Password:", type="password")
        
        if st.button("Login"):
            if username == "Prajen SK" and password == "Prajen@2004":
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Logged in successfully!")
            else:
                st.session_state.authenticated = False
                st.error("Invalid credentials!")
    else:
        st.write(f"Welcome back, {st.session_state.username}")

        # Display chat history
        for message in st.session_state.messages:
            role, content = message['role'], message['content']
            if role == "user":
                st.write(f"**You:** {content}")
            else:
                st.write(f"**Bot:** {content}")

        # Input for user message
        user_input = st.text_input("Your question:", "")

        # File upload
        uploaded_file = st.file_uploader("Upload an image")
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Send"):
            if user_input or image:
                # Analyze image if provided
                image_description = analyze_image(image) if image else ""
                
                # Combine user input with image description
                combined_input = f"{user_input}\n\nImage Analysis: {image_description}"

                # Add combined input to conversation history
                st.session_state.messages.append({"role": "user", "content": combined_input})

                # Get response from Groq API
                response = get_chat_completion(st.session_state.messages)

                # Add bot response to conversation history
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Display the latest interaction
                st.write(f"**You:** {combined_input}")
                st.write(f"**Bot:** {response}")

                # Sentiment analysis
                sentiment = analyze_sentiment(user_input)
                if sentiment > 0:
                    st.write("You seems to be in Good mood, boost up ypur work")
                elif sentiment < 0:
                    st.write("You seems to be in Bad mood, Drink a glass of water.")
                else:
                    st.write("You seem neutral.")

                # Collect feedback
                feedback = st.text_input("Provide feedback (optional):", "")
                if st.button("Submit Feedback"):
                    # Save feedback to a file
                    with open(st.session_state.feedback_file, "a") as f:
                        f.write(f"User: {user_input}\nBot: {response}\nFeedback: {feedback}\n\n")
                    st.write("Thank you for your feedback!")

        # Export chat history
        if st.button("Export Chat History"):
            with open("chat_history.txt", "w") as f:
                for message in st.session_state.messages:
                    role, content = message['role'], message['content']
                    f.write(f"{role.capitalize()}: {content}\n")
            st.write("Chat history exported to 'chat_history.txt'")

        # Multi-language support (dummy example)
        language = st.selectbox("Choose language:", ["English", "Spanish"])
        if language == "Spanish":
            st.write("¡Hola! ¿Cómo puedo ayudarte hoy?")

        # Personalization (dummy example)
        tone = st.radio("Choose chatbot tone:", ["Formal", "Casual"])
        if tone == "Casual":
            st.write("I'll keep it casual!")

        # Scheduled messages (dummy example)
        schedule_message = st.text_input("Schedule a message:", "")
        if st.button("Schedule"):
            st.write(f"Message '{schedule_message}' scheduled!")

        # Interactive buttons
        if st.button("Tell me more"):
            st.write("This innovative Streamlit app combines the power of Groq's chatbot with cutting-edge image analysis. Engage in dynamic conversations with an AI chatbot that understands both your text and visual inputs. Upload images and watch as the bot provides insights based on sophisticated Vision Transformer technology. Enjoy real-time sentiment analysis and provide feedback to enhance your experience. With features like multi-language support and customizable chatbot tones, this app delivers a personalized touch. Export your chat history with ease and explore new ways to interact with AI. Dive into a seamless blend of text and image-based interactions today!")

        # Knowledge base (dummy example)
        if st.button("Show FAQ"):
            st.write("FAQs: 1. How do I get started with the chatbot? 2. What types of images can I upload? 3.How is my feedback used? 4.Can I change the language of the chatbot? 5.What does the sentiment analysis do? 6.How can I export my chat history?")

if __name__ == "__main__":
    main()
