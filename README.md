# Audio Classification Using Machine Learning

This project is an audio classification system that identifies environmental sounds from the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html). Using advanced machine learning and deep learning techniques, the project processes audio data to classify sounds like dog barks, drilling, and more.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Preprocessing and Feature Extraction](#preprocessing-and-feature-extraction)
5. [Model Training](#model-training)
6. [Results](#results)
7. [How to Run the Code](#how-to-run-the-code)
8. [Future Enhancements](#future-enhancements)

---

## Introduction

In the modern world, sound classification is vital for numerous applications, including voice assistants, surveillance, and environmental monitoring. This project builds an end-to-end pipeline for classifying urban sounds by leveraging **Mel-Frequency Cepstral Coefficients (MFCCs)** as features and training a deep learning model.

---

## Dataset

The project uses the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html), which contains labeled sound recordings from 10 different classes, including:

- Dog bark
- Drilling
- Sirens
- Air conditioners, etc.

---

## Exploratory Data Analysis (EDA)

EDA focuses on understanding and visualizing the audio data:

1. **Waveform Visualization**: Using `librosa` and `matplotlib`, the waveform of sounds such as dog barks and drilling was plotted to understand time-domain characteristics.
2. **Metadata Analysis**: The dataset's class distribution was explored to check for class imbalances, which could affect model performance.

---

## Preprocessing and Feature Extraction

### Key Steps:

1. **Audio Reading**:
   - `librosa` was used to read and convert audio files to mono-channel.
   - `scipy` was used for stereo-channel audio comparison.

2. **Feature Extraction**:
   - Extracted **MFCCs (Mel-Frequency Cepstral Coefficients)** to represent audio signals in both time and frequency domains.

3. **Feature Aggregation**:
   - Calculated the mean MFCCs for each audio file to reduce dimensionality while preserving key characteristics.

4. **Data Preparation**:
   - Split the data into training and testing sets (80:20).
   - Encoded class labels into categorical format using one-hot encoding.

---

## Model Training

A **fully connected neural network** was designed and trained using TensorFlow:

- **Architecture**:
  - Three hidden layers with 100, 200, and 100 neurons, respectively.
  - Dropout layers to prevent overfitting.
  - ReLU activation for hidden layers and Softmax activation for the output layer.

- **Training**:
  - Optimized using the Adam optimizer and categorical cross-entropy loss function.
  - Trained for 100 epochs with a batch size of 32.

- **Performance**:
  - Achieved impressive test accuracy, making it reliable for classifying unseen audio samples.

---

## Results

- **Test Accuracy**: The model achieved high accuracy on the test set, showcasing its ability to generalize well.
- **Predictions**: Successfully predicted classes for new audio samples, such as a drilling sound.

Example:
- Input: `drilling_1.wav`
- Prediction: "Drilling"

---

## How to Run the Code

1. **Install Dependencies**:
   ```bash
   pip install librosa matplotlib pandas numpy tensorflow tqdm
   ```

2. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

3. **Prepare the Dataset**:
   - Download the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html).
   - Place the dataset in the project directory.

4. **Run the Notebooks**:
   - Open and run the notebooks:
     - `Audio Classification EDA.ipynb`
     - `Part 2 - Audio Classification Data Preprocessing And Model Creation.ipynb`

5. **Test the Model**:
   - Add any new audio file to the test folder and modify the path in the notebook to predict its class.

---

## Future Enhancements

1. **Convolutional Neural Networks (CNNs)**: Implement CNNs to capture spatial patterns in spectrograms for better performance.
2. **Data Augmentation**: Add techniques like time-stretching, pitch-shifting, and noise injection to enhance model robustness.
3. **Real-time Deployment**: Develop a real-time audio classification app using frameworks like Streamlit or Flask.
4. **More Classes**: Extend the project to classify a broader range of sounds.

---

This project demonstrates a comprehensive approach to audio classification, combining meticulous data analysis, feature engineering, and deep learning. The clear structure and robust methodology make it a perfect showcase for real-world applications in machine learning and AI.

---

Does this look good, or should I refine anything further?
