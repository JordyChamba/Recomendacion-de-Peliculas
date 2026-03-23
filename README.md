# Movie Recommendation System

Un sistema de recomendación de películas profesional implementado en Python, combinando **Collaborative Filtering** y **Content-Based Filtering** con una API REST en FastAPI.

## 🎯 Descripción

Este proyecto implementa tres enfoques de recomendación de películas:

1. **Collaborative Filtering**: Recomienda películas basadas en la similitud entre usuarios
2. **Content-Based Filtering**: Recomienda películas similares en géneros
3. **Enfoque Híbrido**: Combina ambos métodos para mejores resultados

Dataset utilizado: **MovieLens 100K** (100,000 ratings de 943 usuarios sobre 1,682 películas)

## ✨ Características

- ✅ Implementación de dos algoritmos de filtrado complementarios
- ✅ Matriz de similitud coseno para usuarios y películas
- ✅ Evaluación con métricas profesionales (RMSE, MAE, Precision@k, Recall@k)
- ✅ API REST con FastAPI y documentación Swagger
- ✅ Modelo persistente (pickle)
- ✅ Predicción en tiempo real
- ✅ Código modular y bien documentado

## 📸 Capturas del Proyecto

![API Health Check](images/Captura%20desde%202026-03-23%2012-06-24.png)

![API Recomendaciones](images/Captura%20desde%202026-03-23%2012-15-01.png)

![API Stats](images/Captura%20desde%202026-03-23%2012-16-58.png)

![Swagger Documentation](images/Captura%20desde%202026-03-23%2012-17-23.png)

## 📊 Resultados

| Métrica | Valor |
|---------|-------|
| RMSE | ~0.95 |
| MAE | ~0.75 |
| Precision@5 | ~0.62 |
| Recall@5 | ~0.48 |
| Coverage | ~85% |

*Resultados obtenidos en validation set (20% del dataset)*

## 🚀 Quick Start

### Requisitos

- Python 3.8+
- pip

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/tuusuario/movie-recommendation-system.git
cd movie-recommendation-system

# Crear ambiente virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar dataset
python download_data.py
```

### Entrenar el modelo

```bash
python evaluate.py
```

Esto:
1. Descarga el dataset MovieLens 100K (si no existe)
2. Preprocesa y limpia los datos
3. Construye matrices de similitud
4. Evalúa el modelo con métricas profesionales
5. Genera ejemplos de predicción
6. Guarda el modelo en `models/recommendation_model.pkl`

### Iniciar API REST

```bash
# Ejecutar servidor
python -m uvicorn main:app --reload

# Acceder a documentación interactiva
# http://localhost:8000/docs
```

## 📡 API Endpoints

### 1. Obtener recomendaciones para usuario

```bash
GET /users/{user_id}/recommendations?method=hybrid&n_recommendations=5
```

**Parámetros:**
- `user_id`: ID del usuario (1-943)
- `method`: 'collaborative', 'content_based', o 'hybrid' (default: 'hybrid')
- `n_recommendations`: Número de recomendaciones (1-20, default: 5)

**Respuesta:**
```json
{
  "user_id": 1,
  "method": "hybrid",
  "count": 5,
  "recommendations": [
    {
      "movie_id": 127,
      "title": "Godfather, The (1972)",
      "score": 4.523
    },
    {
      "movie_id": 278,
      "title": "Shawshank Redemption, The (1994)",
      "score": 4.421
    }
  ],
  "timestamp": "2024-01-15T10:30:45"
}
```

### 2. Películas similares

```bash
GET /movies/{movie_id}/similar?n_similar=5
```

### 3. Estadísticas del sistema

```bash
GET /stats
```

### 4. Ver todas las películas

```bash
GET /movies?limit=10
```

## 🏗️ Arquitectura del Proyecto

```
movie-recommendation-system/
├── data/                          # Dataset MovieLens
│   ├── u.data                     # Ratings
│   ├── u.item                     # Información de películas
│   └── ...
├── models/                        # Modelos entrenados
│   └── recommendation_model.pkl   # Modelo persistente
├── notebooks/                     # Jupyter notebooks para EDA
│   └── exploratory_analysis.ipynb
├── src/
│   └── recommendation_system.py   # Clase principal del sistema
├── main.py                        # API FastAPI
├── evaluate.py                    # Script de evaluación
├── download_data.py               # Script para descargar datos
├── requirements.txt               # Dependencias
└── README.md
```

## 🔬 Implementación Técnica

### Collaborative Filtering

**Algoritmo:**
1. Construir matriz usuario-película (943 × 1,682)
2. Calcular similitud coseno entre usuarios
3. Para recomendaciones de usuario X:
   - Encontrar usuarios similares a X
   - Identificar películas que ellos vieron pero X no
   - Predecir rating como promedio ponderado por similitud

**Complejidad:** O(n_usuarios²) en preprocesamiento

### Content-Based Filtering

**Algoritmo:**
1. Extraer características de películas (19 géneros binarios)
2. Calcular similitud coseno entre películas
3. Para película Y, retornar películas más similares en géneros

**Complejidad:** O(n_películas²)

### Enfoque Híbrido

Combina ambos métodos con ponderación:
```
score_híbrido = cf_weight * score_cf + (1 - cf_weight) * score_cb
```

## 📈 Evaluación y Métricas

### RMSE (Root Mean Squared Error)
Mide el error en predicción de ratings. Menor es mejor.

### MAE (Mean Absolute Error)
Error absoluto promedio. Más interpretable que RMSE.

### Precision@k
Porcentaje de recomendaciones relevantes en top-k.
```
Precision@5 = (películas relevantes en top-5) / 5
```

### Recall@k
Porcentaje de películas relevantes recuperadas en top-k.
```
Recall@5 = (películas relevantes en top-5) / (total relevantes)
```

### Coverage
Porcentaje del catálogo que puede ser recomendado.

## 💡 Decisiones de Diseño

1. **Similitud Coseno**: Métrica robusta e insensible a magnitud
2. **Matriz Densa**: Aunque los datos son sparse, se utiliza matriz densa para velocidad
3. **Modelo Persistente**: Se guarda el modelo entrenado para no reentrenar cada vez
4. **API REST**: Permite deployar y escalar el sistema fácilmente
5. **Documentación Swagger**: FastAPI genera docs automáticamente

## 🔄 Mejoras Futuras

- [ ] Implementar matrix factorization (SVD, NMF)
- [ ] Deep Learning (neural collaborative filtering)
- [ ] Incorporar contexto temporal
- [ ] A/B testing de algoritmos
- [ ] Monitoreo de performance en producción
- [ ] Cache de recomendaciones frecuentes

## 📚 Referencias

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Collaborative Filtering - Wikipedia](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Recommender Systems - Andrew Ng](https://www.coursera.org/learn/machine-learning-recommenders)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 👤 Autor

[Tu Nombre]
- GitHub: [@tuusuario](https://github.com/tuusuario)
- LinkedIn: [Tu Perfil](https://linkedin.com/in/tuusuario)
- Email: tu@email.com

## 📄 Licencia

MIT License - ver LICENSE.md para detalles

---

## 🙋 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## ⭐ Si te fue útil, considera dar una estrella!
