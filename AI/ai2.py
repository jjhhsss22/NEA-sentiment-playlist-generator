# import necessary modules
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import neattext.functions as nt
import joblib
import numpy as np
from matplotlib import pyplot as plt

# Read dataset
dataset = pd.read_csv("Dataset/one/emotion_sentimen_dataset.csv")
dataset = dataset.drop(columns=["ID"])

# remove stopwords
dataset["Filtered_text"] = dataset["text"].apply(nt.remove_stopwords)

# Split data into features and target
x = dataset["Filtered_text"]
y = dataset["Emotion"]

# Convert labels to one-hot encoded format
y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# vectorize the text into numbers using CV
count_vectorizer = CountVectorizer(max_features=10000)
x_train_CV = count_vectorizer.fit_transform(x_train)
x_test_CV = count_vectorizer.transform(x_test)

# save our count vectorizer
joblib.dump(count_vectorizer, "count_vectorizer2.joblib")


# Convert sparse matrices to SparseTensors
x_train_sparse = tf.sparse.SparseTensor(
    indices=np.array(list(zip(*x_train_CV.nonzero()))),
    values=x_train_CV.data,
    dense_shape=x_train_CV.shape)

x_test_sparse = tf.sparse.SparseTensor(
    indices=np.array(list(zip(*x_test_CV.nonzero()))),
    values=x_test_CV.data,
    dense_shape=x_test_CV.shape)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train_CV.shape[1],)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(13, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train_CV, y_train, epochs=15, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(x_test_CV, y_test)
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
model.save("sentiment_model2.keras")
