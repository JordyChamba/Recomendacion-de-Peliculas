# Guía de Deployment - Movie Recommendation System

Esta guía explica cómo desplegar el sistema de recomendación en **Render** (PaaS gratuito).

## ¿Por qué Render?

- ✅ Tier gratuito generoso
- ✅ Deploy automático desde GitHub
- ✅ Soporte para Python/FastAPI
- ✅ HTTPS incluido
- ✅ No requiere tarjeta de crédito para empezar

## Paso 1: Preparar tu repositorio en GitHub

```bash
# Inicializar repositorio
git init
git add .
git commit -m "Initial commit: Movie Recommendation System"

# Crear repositorio en GitHub y pushear
git remote add origin https://github.com/tuusuario/movie-recommendation-system.git
git branch -M main
git push -u origin main
```

## Paso 2: Crear archivo `render.yaml`

Este archivo le dice a Render cómo construir y ejecutar tu aplicación.

```yaml
services:
  - type: web
    name: movie-recommendation-api
    env: python
    plan: free
    buildCommand: "pip install -r requirements.txt && python download_data.py && python evaluate.py"
    startCommand: "python -m uvicorn main:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
```

## Paso 3: Crear archivo `build.sh` (opcional pero recomendado)

```bash
#!/bin/bash
echo "🔄 Instalando dependencias..."
pip install -r requirements.txt

echo "📥 Descargando datos..."
python download_data.py

echo "🔧 Entrenando modelo..."
python evaluate.py

echo "✅ Build completado"
```

Dale permisos:
```bash
chmod +x build.sh
```

## Paso 4: Pushear cambios a GitHub

```bash
git add render.yaml build.sh
git commit -m "Add Render deployment configuration"
git push origin main
```

## Paso 5: Crear servicio en Render

1. Ir a https://render.com
2. Crear cuenta (gratis)
3. Conectar GitHub
4. Click en "New +" → "Web Service"
5. Seleccionar tu repositorio
6. Configurar:
   - **Name**: `movie-recommendation-api`
   - **Environment**: `Python`
   - **Build Command**: `./build.sh` (o el comando en render.yaml)
   - **Start Command**: `python -m uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
7. Click "Deploy"

## Paso 6: Monitorear el deploy

- Render mostrará logs en tiempo real
- Esperar a que diga "Your service is live"
- Tu API estará en: `https://movie-recommendation-api.onrender.com`

## Probar la API desplegada

```bash
# Reemplazar URL con tu URL de Render
BASE_URL="https://movie-recommendation-api.onrender.com"

# Verificar health
curl $BASE_URL/health

# Obtener recomendaciones
curl "$BASE_URL/users/1/recommendations?n_recommendations=5"

# Ver documentación Swagger
# Abrir en navegador: $BASE_URL/docs
```

## Optimizaciones para Render

### 1. Reducir tamaño del modelo

Si el modelo es muy grande, considerar:
- Usar compresión gzip
- Almacenar en caché
- Usar cloud storage (S3)

### 2. Tiempo de inicio

El tier gratuito de Render puede ser lento. Para mejorar:
- Implementar lazy loading del modelo
- Usar cache de respuestas frecuentes
- Pre-compilar modelos

### 3. Límites de Render (Free Tier)

- 750 horas/mes
- Se apaga después de 15 min de inactividad
- Máximo 0.5 GB RAM
- Máximo 0.25 vCPU

**Solución**: Primera solicitud será lenta (cold start), pero las posteriores serán rápidas.

## Alternativas a Render

### Railway
```bash
railway link
railway up
```

### Heroku (pequeño costo)
```bash
heroku login
heroku create movie-recommendation-api
git push heroku main
```

### AWS Lambda + API Gateway
- Mayor complejidad
- Mejor para escala

### DigitalOcean App Platform
- Pricing: $12/mes
- Mejor control

## Monitoreo en Producción

### Agregar logging

```python
# En main.py
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/users/{user_id}/recommendations")
async def get_recommendations(user_id: int, ...):
    logger.info(f"Recomendaciones solicitadas para usuario {user_id}")
    ...
```

### Agregar métricas

```python
from prometheus_client import Counter, Histogram

recommendation_counter = Counter(
    'recommendations_total', 
    'Total recommendations generated',
    ['method']
)

recommendation_latency = Histogram(
    'recommendation_latency_seconds',
    'Recommendation latency'
)

@app.get("/users/{user_id}/recommendations")
async def get_recommendations(...):
    with recommendation_latency.time():
        ...
    recommendation_counter.labels(method=method).inc()
```

## Mantenimiento

### Actualizar modelo

```bash
# Hacer cambios localmente
git add .
git commit -m "Update recommendation model"
git push origin main

# Render redeploy automáticamente
```

### Monitorear logs

En Render dashboard → Logs

### Escalar si es necesario

- Plan Starter: $7/mes
- Plan Standard: $25/mes

## Seguridad

### Agregar autenticación (opcional)

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/users/{user_id}/recommendations")
async def get_recommendations(
    user_id: int,
    credentials: HTTPAuthCredentials = Depends(security)
):
    # Verificar token
    if credentials.credentials != "tu_token_secreto":
        raise HTTPException(status_code=403, detail="Unauthorized")
    ...
```

### Rate limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/users/{user_id}/recommendations")
@limiter.limit("100/minute")
async def get_recommendations(...):
    ...
```

## Troubleshooting

### Build falla con "ModelNotFound"

```
Error: models/recommendation_model.pkl not found
```

**Solución**: Asegurar que `evaluate.py` se ejecuta en el buildCommand

### API muy lenta

- Primer request: 30-60 segundos (cold start)
- Siguiente requests: 100-500ms

**Solución**: Usar Render Cron Job para "wake up" cada 14 minutos

### Out of Memory

Reducir tamaño del dataset o usar matriz sparse

```python
# En recommendation_system.py
from scipy.sparse import csr_matrix

self.user_movie_matrix = csr_matrix(self.user_movie_matrix)
```

## Recursos

- [Render Docs](https://render.com/docs)
- [FastAPI on Render](https://render.com/docs/deploy-fastapi)
- [Python Environment Variables](https://render.com/docs/environment-variables)
