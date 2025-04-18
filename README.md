# Car Door & Hood Status Detection Web App (Toy Project)

This toy project web application uses a pre-trained deep learning model to detect the status (open/closed) of car doors and hood from uploaded images. It was created as an educational demonstration of computer vision and deep learning concepts.

> **Note:** This project was "vibes coded" using GitHub Copilot Agent with Claude 3.7 Sonnet.

## Features

- **Multi-label Classification**: Simultaneously detects the status of 5 car components:
  - Front Left Door
  - Front Right Door
  - Rear Left Door
  - Rear Right Door
  - Hood
- **User-friendly Interface**: Simple web interface for image upload and result visualization
- **Real-time Processing**: Instant classification results with probability scores
- **Visual Feedback**: Progress bars indicating confidence levels for each component
- **Binary Output**: Provides a binary string representation of all component statuses

## Tech Stack

- **Backend**: Python Flask web server
- **Deep Learning**: PyTorch with ResNet18 architecture
- **Frontend**: HTML, CSS, and JavaScript
- **Image Processing**: PIL (Python Imaging Library)

## Requirements

- Python 3.7+
- Flask 2.3.3
- PyTorch 2.0.0+
- TorchVision 0.15.0+
- Pillow 9.0.0+
- NumPy 1.22.0+

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure the trained model is available in the correct location:
   ```
   model/final_model.pth
   ```

## Usage

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Upload an image of a car using the "Choose Image" button

4. Click "Submit for Classification" to process the image

5. View the results:
   - Binary representation (1 = Open, 0 = Closed)
   - Detailed status of each component with confidence scores

## Project Structure

- `app.py`: Main Flask application with model loading and inference logic
- `templates/index.html`: Web interface for the application
- `model/final_model.pth`: Trained PyTorch model for car component status detection
- `requirements.txt`: Required Python packages

## Model Details

The application uses a fine-tuned ResNet18 model trained on a dataset of car images. The model predicts the status of 5 different car components (4 doors and hood) simultaneously.

- **Architecture**: ResNet18
- **Input Size**: 224×224 pixels
- **Output**: 5 binary classifications (one for each component)
- **Preprocessing**: 
  - Resize to 256px
  - Center crop to 224px
  - Normalize with ImageNet means and standard deviations
  - RGBA images are converted to RGB with white background

## API Endpoints

- `GET /`: Main page with the web interface
- `POST /predict`: Endpoint for image classification
  - Input: Form data with 'file' field containing the image
  - Output: JSON with binary_result and detailed_results