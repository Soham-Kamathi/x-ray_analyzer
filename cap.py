# xray_report_generator_with_text_qa.py

import streamlit as st
from PIL import Image
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from open_clip import create_model_from_pretrained, get_tokenizer
import os
import pdfplumber  # For extracting text from PDF reports

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- Model Loading (with caching) ---
@st.cache_resource
def load_clip_model():
    """
    Loads and caches the BiomedCLIP model and processor for report generation.
    """
    try:
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model.to(device)
        st.success("BiomedCLIP (for report generation) loaded successfully!")
        return preprocess, model, tokenizer
    except Exception as e:
        st.error(f"Error loading BiomedCLIP model: {e}")
        st.info("Please ensure you have a stable internet connection.")
        st.stop()


# --- Report Generation ---
def generate_report(image, processor, model, tokenizer, candidate_labels):
    """
    Generates a descriptive report for the given X-ray image by performing zero-shot classification.
    """
    if not image or not candidate_labels:
        return "Error: Please upload an image and provide at least one descriptive label.", ""

    try:
        template = 'this is a photo of '
        texts = tokenizer([template + label for label in candidate_labels], context_length=256).to(device)

        with torch.no_grad():
            image_processed = processor(image).unsqueeze(0).to(device)
            image_features, text_features, logit_scale = model(image_processed, texts)
            logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)

        probs = logits.cpu().numpy()
        scores = {label: prob for label, prob in zip(candidate_labels, probs[0])}
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # --- Format the Report for UI display and for context ---
        report_for_display = "## X-Ray Analysis Report\n\n**Potential Findings & Confidence Scores:**\n\n"
        report_for_context = "X-Ray Analysis Report. Potential Findings and Confidence Scores are: "

        report_lines = []
        context_lines = []

        for label, score in sorted_scores:
            report_lines.append(f"- **{label}:** {score:.2%}")
            context_lines.append(f"{label} with a confidence of {score:.2%}")

        report_for_display += "\n".join(report_lines)
        report_for_context += ", ".join(context_lines) + "."

        disclaimer = "\n\n---\n**Disclaimer:** This is an AI-generated report. It is **not a substitute for professional medical advice**. Always consult a qualified radiologist."
        report_for_display += disclaimer
        
        return report_for_display, report_for_context

    except Exception as e:
        st.error(f"An error occurred during report generation: {e}")
        return "Failed to generate report.", ""

# --- Compare X-rays ---
def compare_xrays(previous_xray_input, current_xray_input, processor, model, tokenizer, candidate_labels):
    """
    Compares two X-ray images or reports and suggests improvements based on generated reports.
    """
    # If the input is a string, it's assumed to be pre-extracted report text.
    # Otherwise, it's an Image object, and a report needs to be generated.
    previous_report_context = previous_xray_input if isinstance(previous_xray_input, str) else \
                              generate_report(previous_xray_input, processor, model, tokenizer, candidate_labels)[1]
    
    current_report_context = current_xray_input if isinstance(current_xray_input, str) else \
                             generate_report(current_xray_input, processor, model, tokenizer, candidate_labels)[1]

    context = f"Previous report: {previous_report_context}\nCurrent report: {current_report_context}"
    question = "What improvements or advice can be suggested based on these reports? Provide a detailed and insightful analysis as an experienced radiologist."

    return answer_text_question(context, question)

# --- Extract Text from Report ---
def extract_text_from_report(report_file):
    """
    Extracts text from the uploaded report file (PDF or text).
    """
    try:
        if report_file.type == "application/pdf":
            with pdfplumber.open(report_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        elif report_file.type in ["text/plain"]:
            return report_file.read().decode("utf-8").strip()
        else:
            st.error("Unsupported file type for text extraction.")
            return ""
    except Exception as e:
        st.error(f"Error extracting text from report: {e}")
        return ""

# --- Question Answering ---
def answer_text_question(context, question):
    """
    Answers a user's question based on the generated report (context) using the Gemini API.
    """
    if not context or not question:
        return "Cannot answer without a report context and a question."
    
    try:
        from google import genai
        import os  # Import os to access environment variables

        # Get the API key from the environment variable
        api_key = os.getenv("GEMINI_API_KEY")  # Ensure this environment variable is set

        if not api_key:
            raise ValueError("API key is not set in the environment variables.")

        # Initialize the Gemini client with the API key
        client = genai.Client(api_key=api_key)

        # Define the system instruction to set the model's persona
        system_instruction = "You are an experienced radiologist. Analyze the provided X-ray report and answer questions based solely on the information within the report, providing clear and concise medical insights."

        # Generate content using the Gemini API with system instruction
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite", 
            contents=f"Context: {context}\nQuestion: {question}",
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )

        # Return the response text
        return f"**Answer:** {response.text}"

    except Exception as e:
        st.error(f"An error occurred during question answering: {e}")
        return "Sorry, I could not answer the question."


# --- Streamlit User Interface ---
def main():
    """
    The main function to run the Streamlit application.
    """
    st.set_page_config(page_title="X-Ray Q&A", layout="wide")

    # --- Initialize Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "report_context" not in st.session_state:
        st.session_state.report_context = ""
    if "uploaded_image_data" not in st.session_state:
        st.session_state.uploaded_image_data = None
    if "last_uploaded_file_id" not in st.session_state: # To detect new file upload
        st.session_state.last_uploaded_file_id = None

    # --- Sidebar ---
    with st.sidebar:
        st.title("🤖 AI X-Ray Assistant")
        feature_selection = st.selectbox("Select Feature", ["Generate Report & Q&A", "Compare X-Rays"])
        
        st.header("Configuration")
        default_labels = "normal, fracture, pneumonia, cardiomegaly, pleural effusion, nodule, opacity"
        candidate_labels_input = st.text_area(
            "Enter Potential Findings (comma-separated)",
            default_labels,
            height=150,
            help="Provide clinical terms for the initial report generation."
        )
        candidate_labels = [label.strip() for label in candidate_labels_input.split(',') if label.strip()]

        st.markdown("---")
        st.info(
            "**Models Used:**\\n"
            "- **Report Generation:** `BiomedCLIP`\\n"
            "- **Text Q&A:** `Gemini API`"
        )

    # Load models
    clip_processor, clip_model, clip_tokenizer = load_clip_model()

    # --- Main Panel ---
    st.title("X-Ray Report Analysis and Q&A")
    st.markdown("This tool uses **BiomedCLIP** to generate reports from X-rays, then **Gemini API** to answer questions based on those reports.")

    if feature_selection == "Generate Report & Q&A":
        # File uploader for a single X-ray or report
        uploaded_file = st.file_uploader("Choose an X-ray image or report...", type=["png", "jpg", "jpeg", "pdf", "txt"], key="single_xray")

        # Check if a new file has been uploaded to clear previous states
        if uploaded_file is not None and uploaded_file.file_id != st.session_state.last_uploaded_file_id:
            st.session_state.messages = []
            st.session_state.report_context = ""
            st.session_state.uploaded_image_data = None
            st.session_state.last_uploaded_file_id = uploaded_file.file_id
        elif uploaded_file is None and st.session_state.last_uploaded_file_id is not None:
            # If the user clears the uploader
            st.session_state.messages = []
            st.session_state.report_context = ""
            st.session_state.uploaded_image_data = None
            st.session_state.last_uploaded_file_id = None


        if uploaded_file is not None:
            st.header("Uploaded File")
            
            # Initialize current_image here to prevent UnboundLocalError
            current_uploaded_image = None 

            try:
                if uploaded_file.type in ["image/png", "image/jpeg"]:
                    current_uploaded_image = Image.open(uploaded_file).convert("RGB")
                    st.image(current_uploaded_image, caption="Uploaded X-ray", use_column_width=True)
                    st.session_state.uploaded_image_data = current_uploaded_image
                    
                elif uploaded_file.type in ["application/pdf", "text/plain"]:
                    extracted_text = extract_text_from_report(uploaded_file)
                    st.text_area("Extracted Report Text", extracted_text, height=300)
                    st.session_state.report_context = extracted_text  # Use extracted text as context
                    st.session_state.uploaded_image_data = None # Clear image data
                    # Immediately populate chat messages for extracted reports
                    if extracted_text:
                        st.session_state.messages = [{"role": "assistant", "content": "Report text extracted. You can now ask questions."}]
                    else:
                        st.session_state.messages = []
                else:
                    st.error(f"Unsupported file type: {uploaded_file.type}. Please upload an image (png, jpg, jpeg), PDF, or text file.")
                    st.session_state.uploaded_image_data = None
                    st.session_state.report_context = ""
                    st.session_state.messages = []
            except Exception as e:
                st.error(f"Error opening file: {e}")
                st.stop()

            # "Generate Report" Button for images, or to re-display for text reports
            # Only show this button if an image was uploaded, or if a text report was uploaded and
            # we want a explicit "report generated" message for it.
            if st.session_state.uploaded_image_data: # If an image is pending report generation
                if st.button("Generate Report (for Image)", key="generate_button"):
                    if not candidate_labels:
                        st.warning("Please enter at least one potential finding in the sidebar.")
                    else:
                        with st.spinner("Analyzing image and generating report..."):
                            report_display, report_context = generate_report(st.session_state.uploaded_image_data, clip_processor, clip_model, clip_tokenizer, candidate_labels)
                            if report_context:
                                st.session_state.report_context = report_context
                                st.session_state.messages = [{"role": "assistant", "content": report_display}]
                            else:
                                st.error("Could not generate a report to analyze from image.")
                                st.session_state.report_context = ""
                                st.session_state.messages = []
            elif st.session_state.report_context and not st.session_state.messages: # For text reports, just ensure initial message is there
                 if st.button("Confirm Report for Q&A", key="confirm_report_button"):
                    st.session_state.messages = [{"role": "assistant", "content": "Report text confirmed. You can now ask questions."}]
                
            # Display chat interface only if report_context is available
            if st.session_state.report_context:
                st.markdown("---") # Ensure a separator is always there before chat
                
                # Display existing messages
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input for user's question
                if prompt := st.chat_input("Ask a question about the report..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Searching for answer in report..."):
                            response = answer_text_question(st.session_state.report_context, prompt)
                            st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

        else: # No file uploaded yet or cleared
            st.info("Please upload a file (image or report) to begin the analysis.")
            # Clear all relevant states if no file is uploaded
            st.session_state.report_context = ""
            st.session_state.uploaded_image_data = None
            st.session_state.messages = []
            st.session_state.last_uploaded_file_id = None


    elif feature_selection == "Compare X-Rays":
        # File uploaders for two X-rays or reports
        uploaded_file_1 = st.file_uploader("Choose the previous X-ray image or report...", type=["png", "jpg", "jpeg", "pdf", "txt"], key="previous_xray")
        uploaded_file_2 = st.file_uploader("Choose the current X-ray image or report...", type=["png", "jpg", "jpeg", "pdf", "txt"], key="current_xray")

        # Initialize variables to avoid UnboundLocalError
        previous_image = None
        previous_report_text = ""
        current_image = None
        current_report_text = ""

        if uploaded_file_1 is not None and uploaded_file_2 is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.header("Uploaded Previous X-Ray or Report")
                try:
                    if uploaded_file_1.type in ["image/png", "image/jpeg"]:
                        previous_image = Image.open(uploaded_file_1).convert("RGB")
                        st.image(previous_image, caption="Uploaded Previous X-ray", use_column_width=True)
                    elif uploaded_file_1.type in ["application/pdf", "text/plain"]:
                        previous_report_text = extract_text_from_report(uploaded_file_1)
                        st.text_area("Extracted Previous Report Text", previous_report_text, height=300)
                    else:
                        st.error(f"Unsupported file type: {uploaded_file_1.type}. Please upload an image (png, jpg, jpeg), PDF, or text file.")
                except Exception as e:
                    st.error(f"Error opening previous file: {e}")
                    st.stop()

            with col2:
                st.header("Uploaded Current X-Ray or Report")
                try:
                    if uploaded_file_2.type in ["image/png", "image/jpeg"]:
                        current_image = Image.open(uploaded_file_2).convert("RGB")
                        st.image(current_image, caption="Uploaded Current X-ray", use_column_width=True)
                    elif uploaded_file_2.type in ["application/pdf", "text/plain"]:
                        current_report_text = extract_text_from_report(uploaded_file_2)
                        st.text_area("Extracted Current Report Text", current_report_text, height=300)
                    else:
                        st.error(f"Unsupported file type: {uploaded_file_2.type}. Please upload an image (png, jpg, jpeg), PDF, or text file.")
                except Exception as e:
                    st.error(f"Error opening current file: {e}")
                    st.stop()

            # Compare X-Rays Button
            if st.button("Compare X-Rays", key="compare_button"):
                if not candidate_labels:
                    st.warning("Please enter at least one potential finding in the sidebar.")
                else:
                    with st.spinner("Analyzing images and generating reports..."):
                        previous_input_for_comparison = previous_report_text if previous_report_text else previous_image
                        current_input_for_comparison = current_report_text if current_report_text else current_image
                        
                        # Check if either input is still None for images that weren't processed
                        if (uploaded_file_1.type in ["image/png", "image/jpeg"] and previous_input_for_comparison is None) or \
                           (uploaded_file_2.type in ["image/png", "image/jpeg"] and current_input_for_comparison is None):
                           st.error("Please ensure both images are loaded correctly before comparison.")
                        else:
                            comparison_result = compare_xrays(previous_input_for_comparison, current_input_for_comparison, clip_processor, clip_model, clip_tokenizer, candidate_labels)
                            st.markdown(comparison_result)

        else:
            st.info("Please upload both images or reports to begin the analysis.")

if __name__ == "__main__":
    main()