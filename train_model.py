import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt

# 📁 Define dataset path
dataset_dir = r"C:\Users\Shivani\Downloads\merged_dataset"

# 📦 Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation
valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# 🔄 Data generators
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(dataset_dir, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    directory=os.path.join(dataset_dir, 'valid'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 🧠 Build the Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ⚙️ Compile model - Phase 1
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ⏱️ Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

def lr_schedule(epoch, lr):
    return lr if epoch < 5 else lr * 0.5

lr_scheduler = LearningRateScheduler(lr_schedule)

# 🚀 Train model - Phase 1
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    callbacks=[early_stopping, lr_scheduler]
)

# 🔧 Fine-Tuning: Unfreeze last 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# ⚙️ Re-compile for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 🚀 Train model - Phase 2
history_fine = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=15,
    callbacks=[early_stopping, lr_scheduler]
)

# 💾 Save model and plots
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

model_path = os.path.join('models', 'skin_disease_classification_model.h5')
model.save(model_path)
print(f"✅ Model saved to: {model_path}")

# 📊 Accuracy Plot
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history_fine.history['accuracy'], label='Fine-Tuned Training Accuracy')
plt.plot(history_fine.history['val_accuracy'], label='Fine-Tuned Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
acc_plot_path = 'results/accuracy_plot.png'
plt.savefig(acc_plot_path)
plt.show()
print(f"✅ Accuracy plot saved to: {acc_plot_path}")

# 📉 Loss Plot
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history_fine.history['loss'], label='Fine-Tuned Training Loss')
plt.plot(history_fine.history['val_loss'], label='Fine-Tuned Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
loss_plot_path = 'results/loss_plot.png'
plt.savefig(loss_plot_path)
plt.show()
print(f"✅ Loss plot saved to: {loss_plot_path}")
