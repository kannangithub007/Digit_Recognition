
# Handwritten Digit Recognition using MNIST and TensorFlow

## Project Overview

This project is a simple neural network model built with TensorFlow and Keras to recognize handwritten digits (0-9) from images. It uses the popular MNIST dataset, which contains thousands of grayscale images of handwritten digits. The model learns to identify digits by training on this dataset and can predict the digit in new images.

## Features

* Loads and preprocesses the MNIST dataset
* Builds a simple fully connected neural network
* Trains the model for digit classification
* Evaluates model accuracy on test data
* Predicts digits from test images and visualizes the results

## Technologies Used

* Python 3.x
* TensorFlow 2.x
* Keras (built into TensorFlow)
* NumPy
* Matplotlib

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install tensorflow numpy matplotlib
   ```

## How to Run

1. Open the Python script (`mnist_digit_recognition.py`) in your favorite IDE or text editor.

2. Run the script:

   ```bash
   python mnist_digit_recognition.py
   ```

3. The model will:

   * Load and normalize the MNIST data
   * Build and train a neural network for 5 epochs
   * Evaluate the model on the test dataset
   * Display the first test image with actual and predicted digit

## Code Explanation

* **Data Loading:** Uses the built-in `mnist.load_data()` method from Keras to load the dataset.
* **Normalization:** Scales pixel values to \[0, 1] range for better neural network performance.
* **Model Architecture:**

  * `Flatten` layer reshapes each 28x28 image into a 784-element vector.
  * `Dense` layer with 128 neurons and ReLU activation learns features.
  * `Dropout` layer helps prevent overfitting by randomly disabling neurons during training.
  * Final `Dense` layer with 10 neurons outputs probability scores for each digit class using softmax.
* **Training:** Uses Adam optimizer and sparse categorical crossentropy loss function.
* **Evaluation:** Measures accuracy on test data.
* **Prediction and Visualization:** Shows an example digit from the test set with its predicted label.

## Potential Improvements

* Add a graphical interface to allow users to draw their own digits for prediction.
* Use convolutional neural networks (CNNs) for better accuracy.
* Increase training epochs or tune hyperparameters.
* Save and load the trained model for future predictions.

## References

* MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
* TensorFlow Keras Documentation: [https://www.tensorflow.org/api\_docs/python/tf/keras](https://www.tensorflow.org/api_docs/python/tf/keras)

## License

This project is open source and available under the MIT License.


