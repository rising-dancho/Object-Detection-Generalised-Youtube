from tflite_model_maker import object_detector

import config
import tensorflow as tf
import os

# Load config values
EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
MODEL_NAME = config.MODEL_NAME
	@@ -15,6 +18,7 @@
VALID_DATASET_PATH = config.VALID_DATASET_PATH
MODEL = config.MODEL

# Load datasets
train_data = object_detector.DataLoader.from_pascal_voc(
    TRAIN_DATASET_PATH,
    TRAIN_DATASET_PATH,
	@@ -27,8 +31,8 @@
    CLASSES
)

# Create model
spec = model_spec.get(MODEL)
model = object_detector.create(
    train_data,
    model_spec=spec,
	@@ -38,10 +42,32 @@
    validation_data=val_data
)

# Evaluate and export
model.evaluate(val_data)
model.export(export_dir=MODEL_PATH, tflite_filename=MODEL_NAME)

print('-'*100)
print('Training completed.')
print('See the model folder.')

# ✅ Check exported TFLite model output tensor shapes
print("\nAnalyzing output tensor shapes of exported model:")
tflite_path = os.path.join(MODEL_PATH, MODEL_NAME)

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

output_details = interpreter.get_output_details()

found_3d_output = False

for i, out in enumerate(output_details):
    shape = out['shape']
    print(f"Output {i}: name={out['name']}, shape={shape}")

    if len(shape) == 3:
        found_3d_output = True
        print(f"⚠️ Warning: Output {i} is 3D. This may cause ML Kit to throw a dimension error.")

if not found_3d_output:
    print("✅ No 3D output shapes detected. This model is likely safe for ML Kit.")
