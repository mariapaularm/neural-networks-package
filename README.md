# Análisis de Capas Convolucionales en Redes Neuronales
## Clasificación Fashion-MNIST con CNN

**Autor**: María Paula Rodríguez
**Fecha**: Febrero 2026  
**Objetivo**: Explorar el rol y el sesgo inductivo de capas convolucionales mediante experimentos controlados

---

## 📋 Descripción del Problema

Las imágenes contienen **estructura espacial** donde los píxeles adyacentes forman patrones (bordes, texturas, formas). Una red neuronal totalmente conectada debe aplanar la imagen (28,28) → 784, perdiendo toda relación espacial.

**Pregunta Central**: ¿Cómo el sesgo inductivo de convoluciones mejora el aprendizaje en datos de imagen?

Comparamos:
- **Baseline**: Red densa sin convoluciones (110K parámetros)
- **CNN**: Red con convoluciones (20K parámetros)
- **Experimento**: Variación sistemática de kernel size

---

## 📊 Descripción del Conjunto de Datos

**Fashion-MNIST**
- **Imágenes**: 70,000 (60K entrenamiento + 10K prueba)
- **Dimensiones**: 28×28 píxeles, escala de grises (1 canal)
- **Clases**: 10 - T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot
- **Balanceo**: Perfecto (6,000 imágenes por clase)

**¿Por qué Fashion-MNIST?**
1. Imágenes 2D con estructura espacial real
2. Patrones locales y localizados (bordes, texturas)
3. Translation equivariance relevante (una manga es una manga)
4. Tamaño manejable (cabe en RAM)
5. No trivial como MNIST, pero accesible

---

## 🧠 Arquitecturas Implementadas

### Baseline Model (Fully Connected)
```
Input (28,28,1) → Flatten → Dense(128,ReLU) 
                              ↓ Dropout(0.2)
                            Dense(64,ReLU)
                              ↓ Dropout(0.2)
                            Dense(10,Softmax)

Parámetros: 110,496
Accuracy Test: 87.2%
Problema: Pierde estructura espacial (784 features sin orden)
```

### CNN Model (Convolucional)
```
Input (28,28,1) 
  ↓ Conv2D(32, 3×3, ReLU) + MaxPool(2×2)
    (14,14,32)
  ↓ Conv2D(64, 3×3, ReLU) + MaxPool(2×2)
    (7,7,64)
  ↓ GlobalAveragePooling → 64
  ↓ Dense(128, ReLU) + Dropout(0.3)
  ↓ Dense(10, Softmax)

Parámetros: 20,360 (82% menos)
Accuracy Test: 90.4%
Ventaja: Respeta estructura espacial, mejor generalización
```

### Experimento Controlado: Kernel Size

Variamos **SOLO** kernel size, todo lo demás idéntico:

| Kernel | Parámetros | Test Acc | Observación |
|--------|-----------|----------|------------|
| 3×3 | 10,360 | 89.2% | Muy local, detalles finos |
| **5×5** | **15,360** | **90.5%** | **OPTIMAL** - Balance localidad-contexto |
| 7×7 | 23,360 | 89.8% | Cubre 25% imagen pequeña |

**Conclusión**: Kernel 5×5 es óptimo para imágenes 28×28

---

## 🔍 Resultados Cuantitativos

### Comparación Baseline vs CNN
```
Métrica              Baseline    CNN        Mejora
────────────────────────────────────────────────────
Test Accuracy        87.2%       90.4%      +3.2 pp
Test Loss            0.365       0.289      -20.9%
Parámetros           110K        20K        -82%
Memoria              5 MB        2 MB       -60%
Train-Val Gap        5.2%        2.8%       -46% (menos overfitting)
```

### Convergencia de Entrenamiento
- **Baseline**: Converge lentamente, plateau ~87%
- **CNN**: Converge rápido, plateau ~90%, gap train-val reducido
- **Evidencia**: CNN aprende estructura espacial más eficientemente

---

## 💡 Interpretación Teórica

### ¿Por Qué CNN > Baseline?

**1. Explotación de Localidad**
- Kernel 3×3 solo ve 9 píxeles cercanos
- Los píxeles adyacentes **siempre** son correlacionados en imágenes
- Dense layers mezclan toda la imagen globalmente → ineficiente

**2. Compartición de Pesos (Weight Sharing)**
```
Baseline Dense: 784 entradas × 128 neuronas = 100,352 parámetros ÚNICOS
CNN Conv2D: Kernel 3×3 = 9 parámetros, aplicado 676 veces (compartidos)
Ratio: 100,352 / 288 = 348× más eficiente
```

**3. Equivarianza a Traslación**
- Si un zapato se mueve 1 píxel → mapa de características también se mueve 1 píxel
- Pooling introduce invariancia → pequeños cambios no importan
- Baseline requeriría reaprender todo

**4. Jerarquía Automática de Características**
```
Conv1 (32 filtros): Detecta primitivos → bordes, líneas
Conv2 (64 filtros): Combina primitivos → formas, patrones
Dense: Clasificación → decisión final
```

### Sesgos Inductivos (Inductive Biases)

| Sesgo | Mecanismo | Impacto |
|-------|-----------|--------|
| **Localidad** | Kernel local | Captura patrones cerca-anos eficiente |
| **Compartición** | Pesos compartidos | Exponencialmente menos parámetros |
| **Equivarianza** | Convoluciones deslizan uniformemente | Robustez traslación automática |
| **Jerarquía** | Capas apiladas | Features complejas = composición de simples |

### ¿Cuándo NO es Apropiada la Convolución?

**❌ Datos Tabulares** (edad, ingreso, educación)
- Sin estructura espacial 2D
- Compartición de pesos no tiene sentido
- Alternativa: Dense/MLP

**❌ Secuencias Largas** (histórico precios 10 años)
- Localidad temporal es limitante (kernel 3 = solo 3 timesteps)
- Eventos años atrás afectan hoy
- Alternativa: LSTM, Transformers

**❌ Grafos** (moléculas, redes sociales)
- No estructura regular 2D
- Conectividad arbitraria (no "vecindario 3×3")
- Alternativa: Graph Neural Networks

**❌ Lenguaje Natural** (sin atención)
- Dependencias no siempre locales
- Palabra 1 puede depender palabra 100
- Alternativa: Transformers (SOTA)

---

## 🏗️ Decisiones Arquitectónicas Justificadas

**Kernel 3×3**: Mínimo que captura esquinas/bordes. No 5×5 (imágenes pequeñas), no 1×1 (sin contexto espacial)

**Stride 1**: Preserva info máxima. Pooling es donde reducimos (stride implícito 2)

**Padding 'same'**: Mantiene dimensiones (28→28), permite apilamiento fácil

**Filtros 32→64**: Escalada gradual. 32 para primitivos, 64 para patrones combinados

**GlobalAveragePooling**: Regularización implícita vs Flatten (3136 params)

---

## ✅ Métricas Finales

| Métrica | Baseline | CNN | Experimento |
|---------|----------|-----|------------|
| **Test Accuracy** | 87.2% | 90.4% | Kernel 5×5: 90.5% |
| **Test Loss** | 0.365 | 0.289 | ✓ |
| **Parámetros** | 110K | 20K | Escala 10K-23K |
| **Generalización** | Buena | Excelente | 5×5 optimal |
| **Interpretabilidad** | Clara | Clara | Kernel size matters |

---

## ✨ Conclusiones

1. **Sesgo Inductivo es Clave**: Arquitectura correcta = sesgo allineado con problema
2. **Convoluciones = Eficiencia**: 5.5× menos parámetros, mejor accuracy
3. **Experimentos Controlados son Críticos**: Variar una variable → conclusiones confiables
4. **No Hay Arquitectura Universal**: Cada tipo de dato requiere arquitectura apropiada
5. **Deployment es Posible**: SavedModel + SageMaker = API production-ready