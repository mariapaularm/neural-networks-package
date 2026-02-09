
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

def model_fn(model_dir):
    '''
    Carga el modelo desde el directorio especificado.
    SageMaker llamará esta función al iniciar el endpoint.
    '''
    model = keras.models.load_model(f'{model_dir}/1')
    return model

def input_fn(request_body, request_content_type):
    '''
    Parsea la solicitud HTTP y prepara los datos para el modelo.
    Espera JSON con array de imagen (28, 28, 1) normalizado [0,1].
    '''
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        input_data = np.array(data['instances'], dtype=np.float32)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    '''
    Realiza la predicción usando el modelo cargado.
    '''
    predictions = model.predict(input_data)
    return predictions

def output_fn(predictions, content_type):
    '''
    Prepara la salida para enviar al cliente.
    '''
    if content_type == 'application/json':
        # Obtener clase predicha y confianza
        class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
        
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_probs = np.max(predictions, axis=1)
        
        results = []
        for class_id, prob in zip(predicted_classes, predicted_probs):
            results.append({
                'class_id': int(class_id),
                'class_name': class_names[int(class_id)],
                'confidence': float(prob)
            })
        return json.dumps({'predictions': results})
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
