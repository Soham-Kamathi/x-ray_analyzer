# xray_report_generator_with_text_qa.py

import streamlit as st
from PIL import Image
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from open_clip import create_model_from_pretrained, get_tokenizer
import os
import pdfplumber  # For extracting text from PDF reports
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from skimage import measure, morphology
from scipy import ndimage
import io

# Add these anatomical region definitions
ANATOMICAL_REGIONS = {
    'upper_left_lung': {'coords': (0, 0, 0.45, 0.6), 'label': 'Upper Left Lung'},
    'lower_left_lung': {'coords': (0, 0.4, 0.45, 1.0), 'label': 'Lower Left Lung'},
    'upper_right_lung': {'coords': (0.55, 0, 1.0, 0.6), 'label': 'Upper Right Lung'},
    'lower_right_lung': {'coords': (0.55, 0.4, 1.0, 1.0), 'label': 'Lower Right Lung'},
    'heart': {'coords': (0.35, 0.3, 0.65, 0.8), 'label': 'Cardiac Region'},
    'mediastinum': {'coords': (0.4, 0.1, 0.6, 0.9), 'label': 'Mediastinum'},
    'left_costophrenic': {'coords': (0.1, 0.75, 0.4, 0.95), 'label': 'Left Costophrenic Angle'},
    'right_costophrenic': {'coords': (0.6, 0.75, 0.9, 0.95), 'label': 'Right Costophrenic Angle'}
}

def find_activation_regions(cam_map, threshold=0.3, min_area=100):
    """
    Find connected regions of high activation in the CAM map
    """
    # Threshold the CAM map
    binary_map = cam_map > threshold
    
    # Clean up small noise
    binary_map = morphology.remove_small_objects(binary_map, min_size=min_area)
    binary_map = morphology.binary_closing(binary_map, morphology.disk(5))
    
    # Find connected components
    labeled_regions = measure.label(binary_map)
    regions = measure.regionprops(labeled_regions, intensity_image=cam_map)
    
    return regions, labeled_regions

def get_anatomical_region(centroid, image_shape):
    """
    Determine which anatomical region a point belongs to
    """
    y, x = centroid
    h, w = image_shape
    
    # Normalize coordinates to 0-1 range
    norm_x = x / w
    norm_y = y / h
    
    for region_name, region_info in ANATOMICAL_REGIONS.items():
        x1, y1, x2, y2 = region_info['coords']
        if x1 <= norm_x <= x2 and y1 <= norm_y <= y2:
            return region_info['label']
    
    return "Unspecified Region"

def analyze_pathology_regions(segmentation_maps, image_shape, activation_threshold=0.3):
    """
    Analyze segmentation maps to identify specific affected regions
    """
    region_analysis = {}
    
    for pathology, seg_map in segmentation_maps.items():
        regions, labeled_regions = find_activation_regions(seg_map, activation_threshold)
        
        pathology_regions = []
        for i, region in enumerate(regions):
            centroid = region.centroid
            anatomical_location = get_anatomical_region(centroid, image_shape)
            
            region_info = {
                'region_id': i + 1,
                'anatomical_location': anatomical_location,
                'centroid': centroid,
                'area': region.area,
                'max_intensity': region.max_intensity,
                'mean_intensity': region.mean_intensity,
                'bbox': region.bbox  # (min_row, min_col, max_row, max_col)
            }
            pathology_regions.append(region_info)
        
        region_analysis[pathology] = {
            'regions': pathology_regions,
            'labeled_map': labeled_regions
        }
    
    return region_analysis

def create_labeled_overlay_visualization(image, segmentation_maps, region_analysis, alpha=0.4):
    """
    Create visualization with labeled regions and bounding boxes
    """
    try:
        img_array = np.array(image)
        num_pathologies = len(segmentation_maps)
        
        if num_pathologies == 0:
            return None
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, min(num_pathologies, 3), figsize=(18, 12))
        
        # Handle single pathology case
        if num_pathologies == 1:
            axes = axes.reshape(2, 1)
        elif num_pathologies == 2:
            axes = axes.reshape(2, 2)
            
        colors = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
        bbox_colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for idx, (pathology, seg_map) in enumerate(segmentation_maps.items()):
            if idx >= 3:  # Limit to 3 pathologies for display
                break
                
            col_idx = idx % 3
            
            # Top row: Segmentation overlay
            ax_top = axes[0, col_idx] if num_pathologies > 1 else axes[0, 0]
            ax_top.imshow(img_array, cmap='gray')
            ax_top.imshow(seg_map, cmap=colors[idx % len(colors)], alpha=alpha, vmin=0, vmax=1)
            ax_top.set_title(f'{pathology} - Segmentation Map', fontsize=12, fontweight='bold')
            ax_top.axis('off')
            
            # Bottom row: Labeled regions with bounding boxes
            ax_bottom = axes[1, col_idx] if num_pathologies > 1 else axes[1, 0]
            ax_bottom.imshow(img_array, cmap='gray')
            ax_bottom.imshow(seg_map, cmap=colors[idx % len(colors)], alpha=alpha, vmin=0, vmax=1)
            
            # Add region labels and bounding boxes
            if pathology in region_analysis:
                regions = region_analysis[pathology]['regions']
                
                for region in regions:
                    # Draw bounding box
                    min_row, min_col, max_row, max_col = region['bbox']
                    rect = Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                   linewidth=2, edgecolor=bbox_colors[idx % len(bbox_colors)], 
                                   facecolor='none', alpha=0.8)
                    ax_bottom.add_patch(rect)
                    
                    # Add text label
                    centroid_y, centroid_x = region['centroid']
                    label_text = f"Region {region['region_id']}\n{region['anatomical_location']}\nIntensity: {region['max_intensity']:.2f}"
                    
                    ax_bottom.annotate(label_text, 
                                     xy=(centroid_x, centroid_y),
                                     xytext=(10, 10), textcoords='offset points',
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                                     fontsize=8, ha='left', va='bottom',
                                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax_bottom.set_title(f'{pathology} - Labeled Regions', fontsize=12, fontweight='bold')
            ax_bottom.axis('off')
        
        # Hide unused subplots
        for idx in range(num_pathologies, 3):
            if num_pathologies < 3:
                axes[0, idx].axis('off')
                axes[1, idx].axis('off')
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        from PIL import Image as PILImage
        overlay_image = PILImage.open(buf)
        plt.close()
        
        return overlay_image
    
    except Exception as e:
        print(f"Error creating labeled overlay visualization: {e}")
        return None

def generate_region_report(region_analysis):
    """
    Generate a detailed text report of affected regions
    """
    report = "## Detailed Region Analysis\n\n"
    
    if not region_analysis:
        return report + "No significant regions detected.\n"
    
    for pathology, analysis in region_analysis.items():
        regions = analysis['regions']
        
        if not regions:
            report += f"### {pathology}\nNo specific regions identified.\n\n"
            continue
            
        report += f"### {pathology}\n"
        report += f"**Number of affected regions:** {len(regions)}\n\n"
        
        for i, region in enumerate(regions, 1):
            report += f"**Region {i}:**\n"
            report += f"- **Location:** {region['anatomical_location']}\n"
            report += f"- **Size:** {region['area']} pixels\n"
            report += f"- **Maximum Activation:** {region['max_intensity']:.3f}\n"
            report += f"- **Average Activation:** {region['mean_intensity']:.3f}\n"
            
            # Interpret activation levels
            if region['max_intensity'] > 0.8:
                severity = "High confidence detection"
            elif region['max_intensity'] > 0.5:
                severity = "Moderate confidence detection"
            else:
                severity = "Low confidence detection"
            
            report += f"- **Confidence Level:** {severity}\n\n"
    
    report += "---\n**Note:** Region analysis is based on AI model predictions. Clinical correlation is recommended.\n"
    return report

# Updated function to replace the original create_overlay_visualization
def create_enhanced_overlay_visualization(image, segmentation_maps, alpha=0.4, activation_threshold=0.3):
    """
    Enhanced version that includes region labeling
    """
    try:
        img_array = np.array(image)
        image_shape = img_array.shape[:2]
        
        # Analyze regions
        region_analysis = analyze_pathology_regions(segmentation_maps, image_shape, activation_threshold)
        
        # Create labeled visualization
        labeled_overlay = create_labeled_overlay_visualization(image, segmentation_maps, region_analysis, alpha)
        
        # Generate region report
        region_report = generate_region_report(region_analysis)
        
        return labeled_overlay, region_analysis, region_report
    
    except Exception as e:
        print(f"Error in enhanced overlay visualization: {e}")
        return None, {}, "Error generating region analysis."

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

# ChexNet class definitions and labels
CHEXNET_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

class ChexNet(nn.Module):
    """ChexNet model architecture based on DenseNet121"""
    def __init__(self, num_classes=14):
        super(ChexNet, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.densenet121(x)

class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()  # Detach here to avoid gradient tracking
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # Detach here too
        
    def generate_cam(self, input_image, class_idx):
        # Clear previous gradients
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_image)

        # Backward pass
        output[0, class_idx].backward(retain_graph=True)

        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]

        # Compute weights by global average pooling of gradients
        weights = torch.mean(gradients, dim=[1, 2])
        
        # Initialize CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)

        # Compute weighted combination of activation maps
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU and normalize
        cam = torch.relu(cam)
        if torch.max(cam) > 0:  # Avoid division by zero
            cam = cam / torch.max(cam)

        return cam.cpu().numpy()  # Now safe to convert to numpy

@st.cache_resource
def load_chexnet_model():
    """
    Loads and caches the ChexNet model for pathology detection and segmentation.
    """
    try:
        model = ChexNet(num_classes=len(CHEXNET_LABELS))
        # model.load_state_dict(torch.load('chexnet_model.pth.tar', map_location=device))  # Uncomment and set path
        model.to(device)
        model.eval()
        target_layer = model.densenet121.features.denseblock4.denselayer16.conv2
        grad_cam = GradCAM(model, target_layer)
        st.success("ChexNet model loaded successfully!")
        return model, grad_cam
    except Exception as e:
        st.error(f"Error loading ChexNet model: {e}")
        st.info("Please ensure ChexNet model weights are available.")
        model = ChexNet(num_classes=len(CHEXNET_LABELS))
        model.to(device)
        model.eval()
        target_layer = model.densenet121.features.denseblock4.denselayer16.conv2
        grad_cam = GradCAM(model, target_layer)
        return model, grad_cam

def preprocess_image_for_chexnet(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def predict_pathologies(image, chexnet_model, threshold=0.5):
    try:
        image_tensor = preprocess_image_for_chexnet(image)
        with torch.no_grad():
            predictions = chexnet_model(image_tensor)
            predictions = predictions.cpu().numpy()[0]
        results = {}
        for i, label in enumerate(CHEXNET_LABELS):
            results[label] = {
                'probability': float(predictions[i]),
                'detected': predictions[i] > threshold
            }
        return results
    except Exception as e:
        st.error(f"Error in pathology prediction: {e}")
        return {}

def generate_segmentation_map(image, chexnet_model, grad_cam, top_predictions, original_size):
    try:
        image_tensor = preprocess_image_for_chexnet(image)
        segmentation_maps = {}
        for pathology, data in top_predictions.items():
            if data['detected']:
                class_idx = CHEXNET_LABELS.index(pathology)
                cam = grad_cam.generate_cam(image_tensor, class_idx)
                cam_resized = cv2.resize(cam, original_size)
                segmentation_maps[pathology] = cam_resized
        return segmentation_maps
    except Exception as e:
        st.error(f"Error generating segmentation maps: {e}")
        return {}

def create_overlay_visualization(image, segmentation_maps, alpha=0.4):
    try:
        img_array = np.array(image)
        fig, axes = plt.subplots(1, min(len(segmentation_maps) + 1, 4), figsize=(15, 5))
        if len(segmentation_maps) == 0:
            axes = [axes]
        elif not isinstance(axes, np.ndarray):
            axes = [axes]
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        colors = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
        for idx, (pathology, seg_map) in enumerate(segmentation_maps.items()):
            if idx + 1 >= len(axes):
                break
            axes[idx + 1].imshow(img_array, cmap='gray')
            axes[idx + 1].imshow(seg_map, cmap=colors[idx % len(colors)], alpha=alpha, vmin=0, vmax=1)
            axes[idx + 1].set_title(f'{pathology} Segmentation')
            axes[idx + 1].axis('off')
        for idx in range(len(segmentation_maps) + 1, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        from PIL import Image as PILImage
        overlay_image = PILImage.open(buf)
        plt.close()
        return overlay_image
    except Exception as e:
        st.error(f"Error creating overlay visualization: {e}")
        return None

def format_chexnet_results(pathology_results, segmentation_maps):
    try:
        sorted_results = sorted(pathology_results.items(), key=lambda x: x[1]['probability'], reverse=True)
        report = "## ChexNet Pathology Detection Results\n\n"
        detected_pathologies = []
        all_pathologies = []
        for pathology, data in sorted_results:
            prob = data['probability']
            detected = data['detected']
            status = "✅ **DETECTED**" if detected else "❌ Not Detected"
            confidence = f"{prob:.1%}"
            segmentation_note = ""
            if detected and pathology in segmentation_maps:
                segmentation_note = " *(Segmentation Available)*"
            line = f"- **{pathology}:** {status} - Confidence: {confidence}{segmentation_note}"
            all_pathologies.append(line)
            if detected:
                detected_pathologies.append(f"{pathology} ({confidence})")
        if detected_pathologies:
            report += f"**🚨 Detected Pathologies:** {', '.join(detected_pathologies)}\n\n"
        else:
            report += "**✅ No significant pathologies detected**\n\n"
        report += "**Detailed Results:**\n\n"
        report += "\n".join(all_pathologies)
        report += "\n\n---\n**Note:** ChexNet results are AI-generated predictions. Always consult with a qualified radiologist for proper medical diagnosis."
        return report
    except Exception as e:
        st.error(f"Error formatting ChexNet results: {e}")
        return "Error formatting results."

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
        api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Ensure this environment variable is set

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
    if "last_uploaded_file_id" not in st.session_state:
        st.session_state.last_uploaded_file_id = None
    if "chexnet_results" not in st.session_state:
        st.session_state.chexnet_results = {}
    if "segmentation_maps" not in st.session_state:
        st.session_state.segmentation_maps = {}

    # --- Sidebar ---
    with st.sidebar:
        st.title("🤖 AI X-Ray Assistant")
        feature_selection = st.selectbox("Select Feature", [
            "Generate Report & Q&A",
            "ChexNet Pathology Detection",
            "Compare X-Rays"
        ])
        st.header("Configuration")
        default_labels = "normal, fracture, pneumonia, cardiomegaly, pleural effusion, nodule, opacity"
        candidate_labels_input = st.text_area(
            "Enter Potential Findings (comma-separated)",
            default_labels,
            height=150,
            help="Provide clinical terms for the initial report generation."
        )
        candidate_labels = [label.strip() for label in candidate_labels_input.split(',') if label.strip()]
        if feature_selection == "ChexNet Pathology Detection":
            st.subheader("ChexNet Settings")
            detection_threshold = st.slider(
                "Detection Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
                help="Probability threshold for pathology detection"
            )
            
            # New activation threshold for region detection
            activation_threshold = st.slider(
                "Region Activation Threshold",
                min_value=0.1,
                max_value=0.8,
                value=0.3,
                step=0.1,
                help="Threshold for detecting activated regions in segmentation maps"
            )
            
            show_segmentation = st.checkbox(
                "Generate Segmentation Maps",
                value=True,
                help="Create visual segmentation maps for detected pathologies"
            )
            
            # New option for region labeling
            show_region_labels = st.checkbox(
                "Show Region Labels",
                value=True,
                help="Add anatomical region labels to segmentation maps"
            )
            
            show_anatomical_map = st.checkbox(
                "Show Anatomical Regions Map",
                value=False,
                help="Display a reference map of anatomical regions"
            )
        st.markdown("---")
        st.info(
            "**Models Used:**\\n"
            "- **Report Generation:** `BiomedCLIP`\\n"
            "- **Pathology Detection:** `ChexNet`\\n"
            "- **Text Q&A:** `Gemini API`"
        )

    # Load models
    clip_processor, clip_model, clip_tokenizer = load_clip_model()
    chexnet_model, grad_cam = load_chexnet_model()

    # --- Main Panel ---
    st.title("X-Ray Report Analysis and Q&A")
    st.markdown("This tool uses **BiomedCLIP** to generate reports from X-rays, **ChexNet** for pathology detection and segmentation, and **Gemini API** to answer questions based on those reports.")

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


    elif feature_selection == "ChexNet Pathology Detection":
        st.header("ChexNet Pathology Detection & Segmentation")
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"], key="chexnet_xray")
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="Uploaded X-ray", use_column_width=True)
                    
                    # Show anatomical regions map if requested
                    if show_anatomical_map:
                        with st.spinner("Generating anatomical regions map..."):
                            anatomical_map = create_interactive_region_map(np.array(image).shape[:2])
                            st.image(anatomical_map, caption="Anatomical Regions Reference", use_column_width=True)
                
                with col2:
                    if st.button("Analyze with ChexNet", key="chexnet_analyze"):
                        with st.spinner("Running ChexNet pathology detection..."):
                            pathology_results = predict_pathologies(image, chexnet_model, detection_threshold)
                            st.session_state.chexnet_results = pathology_results
                            
                            if show_segmentation and pathology_results:
                                detected_pathologies = {k: v for k, v in pathology_results.items() if v['detected']}
                                
                                if detected_pathologies:
                                    with st.spinner("Generating segmentation maps..."):
                                        segmentation_maps = generate_segmentation_map(
                                            image, chexnet_model, grad_cam,
                                            detected_pathologies, image.size
                                        )
                                        st.session_state.segmentation_maps = segmentation_maps
                                        
                                        # Generate region analysis if labeling is enabled
                                        if show_region_labels and segmentation_maps:
                                            with st.spinner("Analyzing affected regions..."):
                                                labeled_overlay, region_analysis, region_report = create_enhanced_overlay_visualization(
                                                    image, segmentation_maps, alpha=0.4, activation_threshold=activation_threshold
                                                )
                                                st.session_state.labeled_overlay = labeled_overlay
                                                st.session_state.region_analysis = region_analysis
                                                st.session_state.region_report = region_report
                                
                # Display results
                if st.session_state.chexnet_results:
                    st.markdown("---")
                    
                    # Display standard ChexNet results
                    results_report = format_chexnet_results(
                        st.session_state.chexnet_results,
                        st.session_state.segmentation_maps
                    )
                    st.markdown(results_report)
                    
                    # Display segmentation visualizations
                    if st.session_state.segmentation_maps and show_segmentation:
                        st.subheader("Pathology Segmentation Maps")
                        
                        # Choose which visualization to show
                        if show_region_labels and hasattr(st.session_state, 'labeled_overlay') and st.session_state.labeled_overlay:
                            st.image(st.session_state.labeled_overlay, caption="Enhanced Segmentation with Region Labels", use_column_width=True)
                            
                            # Display detailed region analysis
                            if hasattr(st.session_state, 'region_report'):
                                st.markdown(st.session_state.region_report)
                                
                            # Create expandable section for technical details
                            with st.expander("Technical Region Analysis Details"):
                                if hasattr(st.session_state, 'region_analysis'):
                                    for pathology, analysis in st.session_state.region_analysis.items():
                                        st.write(f"**{pathology}:**")
                                        for region in analysis['regions']:
                                            st.json(region)
                        else:
                            # Fallback to standard visualization
                            overlay_image = create_overlay_visualization(
                                image, st.session_state.segmentation_maps
                            )
                            if overlay_image:
                                st.image(overlay_image, caption="Standard Segmentation Overlays", use_column_width=True)
                    
                    # Download options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="Download ChexNet Report",
                            data=results_report,
                            file_name="chexnet_analysis_report.md",
                            mime="text/markdown"
                        )
                    
                    with col2:
                        if show_region_labels and hasattr(st.session_state, 'region_report'):
                            st.download_button(
                                label="Download Region Analysis",
                                data=st.session_state.region_report,
                                file_name="region_analysis_report.md",
                                mime="text/markdown"
                            )
                    
                    with col3:
                        if hasattr(st.session_state, 'region_analysis'):
                            import json
                            analysis_json = json.dumps(st.session_state.region_analysis, indent=2, default=str)
                            st.download_button(
                                label="Download Technical Data",
                                data=analysis_json,
                                file_name="region_analysis_data.json",
                                mime="application/json"
                            )
                            
            except Exception as e:
                st.error(f"Error processing image: {e}")
        else:
            st.info("Please upload an X-ray image to begin ChexNet analysis.")
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
