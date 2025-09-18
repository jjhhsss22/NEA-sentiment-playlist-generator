# import necessary modules
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import neattext.functions as nt

# Read dataset
train_dataset = pd.read_csv("Dataset/two/train.txt", sep=';', names=["Text", "Emotion"])
test_dataset = pd.read_csv("Dataset/two/test.txt", sep=';', names=["Text", "Emotion"])

# remove stopwords
train_dataset["Filtered_text"] = train_dataset["Text"].apply(nt.remove_stopwords)
test_dataset["Filtered_text"] = test_dataset["Text"].apply(nt.remove_stopwords)

# Split data into features and target
x_train = train_dataset["Filtered_text"]
y_train = train_dataset["Emotion"]
x_test = test_dataset["Filtered_text"]
y_test = test_dataset["Emotion"]

# Convert labels to one-hot encoded format
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# vectorize the text into numbers using tfidf
CountVectorizer = CountVectorizer(max_features=10000)
x_train_CV = CountVectorizer.fit_transform(x_train)
x_test_CV = CountVectorizer.fit_transform(x_test)

# Pad sequences to ensure uniform length
max_length = x_train_CV.shape[1]
x_train_padded = pad_sequences(x_train_CV.toarray(), maxlen=max_length, padding='post')
x_test_padded = pad_sequences(x_test_CV.toarray(), maxlen=max_length, padding='post')

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train_padded, y_train, epochs=50, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(x_test_padded, y_test)
print("Accuracy: ", accuracy)
print("Loss: ", loss)

# Save the model
model.save("sentiment_model.keras")
