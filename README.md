# EcoWasteClassifier


The Waste Classifier Sustainability App is an intelligent web application that classifies images of waste into categories using a pre-trained machine learning model. In this app, we demonstrate how to use the low-code tool Teachable Machine to train an AI/ML model for image classification. Specifically, we create a waste classifier that distinguishes between various waste categories such as Cardboard, Glass, Metal,  Plastic

Once the model is trained, we integrate OpenAI's GPT-3 to generate insights about the carbon emissions associated with each predicted waste class. In addition, the app presents relevant United Nations Sustainable Development Goals (SDGs) to further raise awareness about the environmental impact of waste management.

All of this functionality is integrated into a user-friendly web app built with Streamlit, allowing for real-time predictions and providing users with immediate interaction with the waste classification model.

## Features

- **Waste Classification**: Classify images into categories such as Cardboard, Plastic, Glass, and Metal.
- **Carbon Footprint Information**: Get an approximation of the carbon footprint for the waste category, generated using **OpenAI's GPT model**.
- **Sustainable Development Goals (SDGs)**: Display related SDG goals based on the classification of the waste.

## Dataset

The model was trained using the [Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) from Kaggle. This dataset contains images of various waste categories to train the machine learning model for accurate classification.

## Model Training

The model used in this app was created using **Teachable Machine**, a user-friendly platform by Google that allows users to train custom machine learning models. The platform simplifies the model creation process, which was then deployed to classify images of waste into predefined categories.

## Requirements

To run this project locally, you need the following Python dependencies:

- `tensorflow-cpu` (for environments without a GPU)
- `numpy`
- `Pillow`
- `streamlit`
- `google-generativeai`
- `python-dotenv`

You can install these dependencies by running:

```bash
pip install -r requirements.txt
```

## Setup

1. **Clone the repository** to your local machine:

```bash
git clone https://github.com/PriyanshuDey23/EcoWasteClassifier.git
```

2. **Set up environment variables**:
   Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

3. **Model and Labels**:
   - Ensure the **Keras model file** (`keras_model.h5`) is in the same directory as the script.
   - Ensure you have the **labels.txt** file containing the class names of the waste categories.

4. **Run the app**:
   Start the Streamlit app by running:

```bash
streamlit run app.py
```

This will launch the app in your default web browser, allowing you to upload images and classify them.

## How it Works

1. **Image Classification**:
   - Upload an image of waste (such as Cardboard, Plastic, Glass, or Metal).
   - The app will preprocess the image and pass it through a pre-trained machine learning model to classify the waste.
   - The model's prediction is displayed along with the confidence score.

2. **Carbon Footprint Information**:
   - Based on the classification label, the app queries OpenAI's GPT model to provide an approximate carbon footprint for the specific waste category.
   - The result includes a brief description of the environmental impact of the waste type.

3. **Sustainable Development Goals (SDGs)**:
   - Relevant SDG images are displayed, providing a visual connection between waste management and global sustainability efforts.

## File Structure

```
/project_directory
│
├── app.py                  # Streamlit app entry point
├── keras_model.h5          # Pre-trained Keras model
├── labels.txt              # Waste class labels
├── .env                    # Environment variables for OpenAI API key
├── sdg_goals/              # Folder containing SDG images
│
└── requirements.txt        # List of dependencies
```

## Usage

1. Upload an image of waste (Cardboard, Plastic, Glass, or Metal).
2. The app will classify the waste and display the predicted category.
3. The app will show the corresponding SDG goals and carbon footprint information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
