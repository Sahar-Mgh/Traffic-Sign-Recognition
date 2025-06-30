import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import os
import zipfile
import tempfile



# Define paths 
train_dir = r'C:\Users\sahar\Desktop\DL exercise\Data\Train'
test_dir = r'C:\Users\sahar\Desktop\DL exercise\Data\Test'
test_csv = r'C:\Users\sahar\Desktop\DL exercise\Data\Test.csv'

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32


# Load datasets without initial mapping
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

class_names = train_ds.class_names
num_classes = len(class_names)

# Define data augmentation layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

#  apply augmentation AND mobilenet preprocessing
def augment_and_preprocess(image, label):
    image = data_augmentation(image, training=True)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

# Apply functions to the dataset
train_ds = train_ds.map(augment_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
# validation data
val_ds = val_ds.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)

# Prefetch data for performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Create the base model from the pre-trained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model layers
base_model.trainable = False

# C full model
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Compile 
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train 
epochs = 10
history = model.fit(train_ds,
                    epochs=epochs,
                    validation_data=val_ds)

model.save('traffic_sign_model_full.h5')



prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

#  aiming for 60% sparsity to preserve accuracy.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.25,
                                                               final_sparsity=0.60,
                                                               begin_step=0,
                                                               end_step=np.ceil(len(train_ds)).astype(np.int32) * 10)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# Better Fine-Tuning 
optimizer_for_pruning = tf.keras.optimizers.Adam(learning_rate=0.0001)

model_for_pruning.compile(optimizer=optimizer_for_pruning,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                          metrics=['accuracy'])

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
]

model_for_pruning.fit(train_ds,
                      epochs=10, 
                      validation_data=val_ds,
                      callbacks=callbacks)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.save('traffic_sign_model_pruned.h5')

# The model is now trained with pruning.
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.save('traffic_sign_model_pruned.h5')



# Load the pruned Keras model
pruned_keras_model = tf.keras.models.load_model('traffic_sign_model_pruned.h5')

# Convert to a standard TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_keras_model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = 'traffic_sign_model_pruned.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# create a quantized TFLite model for max compression
def representative_data_gen():
  for input_value, _ in train_ds.take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(pruned_keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_quant_model = converter.convert()
tflite_quant_model_path = 'traffic_sign_model_pruned_quant.tflite'
with open(tflite_quant_model_path, 'wb') as f:
    f.write(tflite_quant_model)



# Load the test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False) # No need to shuffle for evaluation

# Apply the same preprocessing as the validation set
test_ds = test_ds.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
                      num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Load your final models
full_model = tf.keras.models.load_model('traffic_sign_model_full.h5')
pruned_model = tf.keras.models.load_model('traffic_sign_model_pruned.h5')

print("\n--- Final Model Performance on Unseen Test Data ---")

# Evaluate the original, full model
loss_full, accuracy_full = full_model.evaluate(test_ds)
print(f"Original Full Model -> Test Accuracy: {accuracy_full*100:.2f}%")

# Evaluate the final, pruned model
loss_pruned, accuracy_pruned = pruned_model.evaluate(test_ds)
print(f"Final Pruned Model -> Test Accuracy: {accuracy_pruned*100:.2f}%")




# final Comparison
print("\n--- Model Comparison ---")
full_model_size = os.path.getsize('traffic_sign_model_full.h5') / (1024 * 1024)
pruned_tflite_size = os.path.getsize(tflite_model_path) / (1024 * 1024)
quant_tflite_size = os.path.getsize(tflite_quant_model_path) / (1024 * 1024)

print(f"Original Full Model Size (H5): {full_model_size:.2f} MB")
print(f"Pruned TFLite Model Size: {pruned_tflite_size:.2f} MB")
print(f"Pruned & Quantized TFLite Model Size: {quant_tflite_size:.2f} MB\n")

reduction_percentage = (1 - (quant_tflite_size / full_model_size)) * 100
print(f"Total size reduction vs original: {reduction_percentage:.2f}%")
