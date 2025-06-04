# Transformer-Based Sentiment Analysis Model

This project implements a transformer-based sentiment analysis model using the IMDB dataset. The model is built using TensorFlow and includes data preprocessing, tokenization, and a custom transformer architecture.

## Project Structure

- **Transformer_based_sentiment_analysis_model.ipynb**: Jupyter Notebook containing the code for data preprocessing, model building, training, and evaluation.
- **IMDB Dataset.csv**: Dataset used for training and testing the sentiment analysis model.
- **transformer_sentiment_model.h5**: Saved model file after training.

## Steps to Run the Project

1. **Install Required Libraries**:
   Ensure you have the following Python libraries installed:
   - pandas
   - numpy
   - matplotlib
   - nltk
   - scikit-learn
   - tensorflow

   You can install them using pip:
   ```bash
   pip install pandas numpy matplotlib nltk scikit-learn tensorflow
   ```

2. **Download the Dataset**:
   Place the `IMDB Dataset.csv` file in the project directory.

3. **Run the Notebook**:
   Open the `Transformer_based_sentiment_analysis_model.ipynb` notebook in Jupyter Notebook or VS Code and execute the cells sequentially.

4. **Train the Model**:
   The notebook includes steps to preprocess the data, build the transformer model, and train it on the IMDB dataset.

5. **Evaluate the Model**:
   Evaluate the model's performance using the test dataset. The notebook provides accuracy and classification reports.

6. **Save and Load the Model**:
   The trained model is saved as `transformer_sentiment_model.h5`. You can load this model for future predictions.

7. **User Input Prediction**:
   The notebook includes a function to predict the sentiment of user-provided reviews.

## Key Features

- **Data Preprocessing**: Includes cleaning, tokenization, and padding of text data.
- **Transformer Architecture**: Custom implementation of a transformer-based model for sentiment analysis.
- **Visualization**: Plots training and validation accuracy and loss.
- **User Interaction**: Allows users to input reviews and get sentiment predictions.

## Example Usage

1. Enter a review in the input prompt:
   ```
   Enter your Review here: The movie was fantastic and very engaging!
   ```
2. Get the predicted sentiment:
   ```
   Predicted Sentiment: Positive
   ```

## References

- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
