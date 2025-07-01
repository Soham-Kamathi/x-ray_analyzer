# X-Ray Report Generator and Q&A

## Overview
This project is an AI-powered application that utilizes the BiomedCLIP model for generating reports from X-ray images and the Gemini API for answering questions based on those reports. The application is built using Streamlit, allowing users to interactively upload X-ray images or reports and receive insights.

## Features
- Generate descriptive reports from X-ray images.
- Compare two X-ray images or reports and suggest improvements.
- Answer questions based on the generated reports using the Gemini API.
- Support for both image and text/PDF report uploads.

## Requirements
- Python 3.7 or higher
- Streamlit
- PyTorch
- Transformers
- OpenCLIP
- pdfplumber
- Google GenAI SDK

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/x-ray_analyzer.git
   cd x-ray_analyzer
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   - Create a `.env` file in the project root and add your API key using the following commands:
     ```bash
     echo "GEMINI_API_KEY=your_api_key_here" > .env
     ```

## Usage
1. Run the application:
   ```bash
   streamlit run cap.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Use the sidebar to select features:
   - **Generate Report & Q&A**: Upload an X-ray image or report to generate a report and ask questions.
   - **Compare X-Rays**: Upload two X-ray images or reports to compare and receive suggestions.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.



## Acknowledgments
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP) for the report generation model.
- [Gemini API](https://cloud.google.com/gemini) for the question-answering capabilities.