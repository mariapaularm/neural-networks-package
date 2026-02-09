# Explorando capas convolucionales a través de datos y experimentos

## Descripción del problema
Las imágenes contienen estructura espacial, ya que los píxeles cercanos forman patrones como bordes y formas que son importantes para reconocer objetos. Sin embargo, una red neuronal totalmente conectada necesita aplanar la imagen y tratar cada píxel como una entrada independiente, lo que hace que se pierda gran parte de esta información espacial.

En este proyecto se compara un modelo base sin capas convolucionales con una red neuronal convolucional (CNN) para la tarea de clasificación de imágenes. El objetivo es analizar cómo el uso de capas convolucionales y algunas decisiones simples de arquitectura influyen en el desempeño del modelo, usando el conjunto de datos Fashion-MNIST como caso de estudio.

## Descripción del conjunto de datos
Fashion-MNIST es un conjunto de datos de imágenes en escala de grises de prendas de vestir, organizado en 10 clases diferentes, como camisetas, pantalones, zapatos y bolsos. Cada imagen tiene un tamaño de 28×28 píxeles.

El conjunto de datos está dividido en un conjunto de entrenamiento y uno de prueba, y es ampliamente utilizado como referencia para evaluar modelos de clasificación de imágenes. Debido a que las imágenes contienen patrones visuales locales y una estructura espacial clara, Fashion-MNIST es adecuado para experimentar con redes neuronales convolucionales.

## Diagramas de arquitectura

## Resultados experimentales

## Interpretación 