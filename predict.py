# from PIL import Image
# import numpy as np
# import tensorflow as tf

# def predict(image_path, model):
#     """
#     Predict the class using the provided model.
#     """
#     # Load the image
#     img = Image.open(image_path).resize((224, 224))  # Ensure the image matches your model's input shape
#     img_array = np.array(img) / 255.0  # Normalize image
    
#     # Reshape image if required by your model (e.g., for a CNN model)
#     img_array = img_array.reshape(1, 224, 224, 3)  # Adjust shape based on your model's input requirements
    
#     # Make prediction using the loaded model
#     predictions = model.predict(img_array)
    
#     # Define your class labels (disease names)
#     class_labels = ["Healthy", "Glaucoma", "Cataract", "Diabetic Retinopathy"]
    
#     # Process prediction result
#     predicted_class = np.argmax(predictions, axis=1)[0]  # Adjust based on your model's output
#     confidence = predictions[0][predicted_class] * 100

#     # Return result with disease names
#     return {
#         "primaryDiagnosis": class_labels[predicted_class],  # Use the disease name instead of the class number
#         "primaryConfidence": confidence,
#         "otherDiagnoses": [
#             {"name": class_labels[i], "confidence": float(predictions[0][i] * 100)} 
#             for i in range(len(predictions[0]))  # Adjust based on your model's output
#         ]
#     }

from PIL import Image
import numpy as np
import tensorflow as tf

def predict(image_path, model):
    """
    Predict the class using the provided model.
    """
    # Load the image
    img = Image.open(image_path).resize((224, 224))  # Resize the image to 224x224
    
    # Ensure the image has 3 channels (RGB), if it's grayscale, convert it
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert image to numpy array and normalize the pixel values (between 0 and 1)
    img_array = np.array(img) / 255.0  # Normalize image values
    
    # Reshape image to match the model input (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction using the loaded model
    predictions = model.predict(img_array)
    
    # Process prediction result
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the predicted class
    confidence = predictions[0][predicted_class] * 100  # Get the confidence of the prediction

    # Return result with the specific disease names
    class_names = ["Healthy", "Glaucoma", "Cataract", "Diabetic Retinopathy"]
    
    return {
        "primaryDiagnosis": class_names[predicted_class],
        "primaryConfidence": confidence,
        "otherDiagnoses": [
            {"name": class_names[i], "confidence": float(predictions[0][i] * 100)}
            for i in range(len(predictions[0]))  # Iterate through all possible classes
        ]
    }
