import numpy as np
import cv2

# Chargement du modèle TFLite
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

MODEL_PATH = "/home/nicaise/Bureau/cyber_agent/projet licence/model/MobileFaceNet.tflite"

# Charge le modèle
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Inspecte les entrées/sorties attendues par TON modèle spécifique
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== TON MODELE MOBILEFACENET ===\n")
print("ENTRÉE attendue :")
print(f"  Shape : {input_details[0]['shape']}")
print(f"  Type  : {input_details[0]['dtype']}")

print("\nSORTIE produite :")
print(f"  Shape : {output_details[0]['shape']}")
print(f"  Type  : {output_details[0]['dtype']}")