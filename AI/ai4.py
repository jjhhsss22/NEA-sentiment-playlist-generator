# import necessary modules
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import neattext.functions as nt
import joblib
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from matplotlib import pyplot as plt

# Read dataset
dataset = pd.read_csv("Dataset/one/emotion_sentimen_dataset.csv")
dataset = dataset.drop(columns=["ID"])

# remove stopwords
dataset["text"] = dataset["text"].apply(nt.remove_stopwords)

# porter stemmer
ps = PorterStemmer()


def preprocess_text(text):
    words = word_tokenize(text)
    stemmed_words = [ps.stem(word) for word in words]

    return ' '.join(stemmed_words)


# Apply the preprocessing function to the text column
dataset['Filtered_text'] = dataset['text'].apply(preprocess_text)

x = dataset['Filtered_text']
y = dataset["Emotion"]

# Convert labels to one-hot encoded format
y = pd.get_dummies(y)

# count vectorizer
count_vectorizer = CountVectorizer()

x = count_vectorizer.fit_transform(x)  # max_features=10000

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# save our count vectorizer
joblib.dump(count_vectorizer, "count_vectorizer4.joblib")

# # Convert sparse matrices to SparseTensors
# x_train_sparse = tf.sparse.SparseTensor(
#     indices=np.array(list(zip(*x_train_CV.nonzero()))),
#     values=x_train_CV.data,
#     dense_shape=x_train_CV.shape)
#
# x_test_sparse = tf.sparse.SparseTensor(
#     indices=np.array(list(zip(*x_test_CV.nonzero()))),
#     values=x_test_CV.data,
#     dense_shape=x_test_CV.shape)

# Define the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(32, activation='relu'),
    # tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(13, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy: ", accuracy)
print("Loss: ", loss)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('model train vs validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.show()

# Save the model
model.save("sentiment_model4.keras")
