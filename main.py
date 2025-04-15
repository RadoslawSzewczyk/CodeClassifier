import os
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime

data_dir = "trainingData"

samples = []
labels = []

for filename in os.listdir(data_dir):
    if filename.endswith('.txt'):
        label = filename.split('.')[0]
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            samples.append(text)
            labels.append(label)

data = list(zip(samples, labels))
random.shuffle(data)
samples, labels = zip(*data)

unique_labels = sorted(list(set(labels)))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
numeric_labels = [label_to_index[label] for label in labels]

split_index = int(0.8 * len(samples))
train_texts = samples[:split_index]
train_labels = numeric_labels[:split_index]
test_texts = samples[split_index:]
test_labels = numeric_labels[split_index:]

max_features = 10000
sequence_length = 250
embedding_dim = 64

vectorize_layer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)
vectorize_layer.adapt(train_texts)

batch_size = 32
train_ds = tf.data.Dataset.from_tensor_slices((list(train_texts), list(train_labels)))
train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((list(test_texts), list(test_labels)))
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Build the model using the Sequential API
# The model consists of:
#   1. The vectorization layer to convert text to integer sequences.
#   2. An Embedding layer.
#   3. A 1D convolutional layer.
#   4. Global max pooling.
#   5. A couple of dense layers with a softmax output.
model = models.Sequential([
    vectorize_layer,
    layers.Embedding(max_features, embedding_dim, mask_zero=True),
    layers.Conv1D(64, 5, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(unique_labels), activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

epochs = 10
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs
)

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

model.save('code_classifier_model')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_filename = f"results_{timestamp}.txt"

with open(results_filename, 'w') as file:
    file.write("Model Evaluation Results\n")
    file.write("========================\n")
    file.write(f"Test Loss: {test_loss}\n")
    file.write(f"Test Accuracy: {test_accuracy}\n\n")
    file.write("Training History:\n")
    file.write(str(history.history))
    
print(f"Results saved to {results_filename}")
