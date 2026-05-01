import tensorflow as tf
import os

model_path = 'vgg16_final.h5'
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("TOP LEVEL LAYERS:")
        for l in model.layers:
            print(f"- {l.name}")
        
        if 'vgg16' in [l.name for l in model.layers]:
            vgg = model.get_layer('vgg16')
            print("\nVGG LAYERS:")
            for l in vgg.layers:
                print(f"  - {l.name}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file {model_path} not found.")
