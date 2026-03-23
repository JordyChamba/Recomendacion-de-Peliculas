"""
API REST para el Sistema de Recomendación de Películas
Usa FastAPI para crear endpoints profesionales y documentados
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from src.recommendation_system import MovieRecommendationSystem

# Inicializar aplicación
app = FastAPI(
    title="Movie Recommendation System API",
    description="API para recomendaciones de películas con Collaborative y Content-Based Filtering",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de datos
class RecommendationResponse(BaseModel):
    movie_id: int
    title: str
    score: float

class RecommendationsListResponse(BaseModel):
    user_id: int
    method: str
    recommendations: List[RecommendationResponse]
    timestamp: str

class MovieSimilarityResponse(BaseModel):
    movie_id: int
    title: str
    similarity_score: float

# Cargar modelo al iniciar
recommendation_system = None

@app.on_event("startup")
async def load_model():
    """Carga el modelo al iniciar la API"""
    global recommendation_system
    
    model_path = 'models/recommendation_model.pkl'
    
    if not os.path.exists(model_path):
        print("⚠️  Modelo no encontrado. Entrenando...")
        recommendation_system = MovieRecommendationSystem()
        recommendation_system.train()
    else:
        print("📂 Cargando modelo...")
        recommendation_system = MovieRecommendationSystem()
        recommendation_system.load_model(model_path)
    
    print("✓ Modelo cargado exitosamente")


@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "name": "Movie Recommendation System",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Verificar estado de la API",
            "GET /movies": "Obtener lista de películas",
            "GET /users/{user_id}/recommendations": "Obtener recomendaciones para usuario",
            "GET /movies/{movie_id}/similar": "Obtener películas similares",
            "POST /recommendations": "Obtener recomendaciones con parámetros personalizados",
            "GET /stats": "Estadísticas del sistema"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Verifica que la API esté funcionando"""
    return {
        "status": "healthy",
        "model_loaded": recommendation_system is not None
    }


@app.get("/movies")
async def get_all_movies(limit: Optional[int] = 100):
    """
    Obtiene la lista de películas disponibles
    
    - **limit**: Máximo número de películas a retornar
    """
    movies_list = recommendation_system.movies_df[['movie_id', 'title']].head(limit).to_dict('records')
    return {
        "total": len(recommendation_system.movies_df),
        "returned": len(movies_list),
        "movies": movies_list
    }


@app.get("/users/{user_id}/recommendations")
async def get_user_recommendations(
    user_id: int,
    method: str = "hybrid",
    n_recommendations: int = 5
):
    """
    Obtiene recomendaciones para un usuario específico
    
    - **user_id**: ID del usuario
    - **method**: Tipo de filtrado: 'collaborative', 'content_based', o 'hybrid'
    - **n_recommendations**: Número de recomendaciones (1-20)
    """
    
    if n_recommendations < 1 or n_recommendations > 20:
        raise HTTPException(status_code=400, detail="n_recommendations debe estar entre 1 y 20")
    
    if method not in ["collaborative", "content_based", "hybrid"]:
        raise HTTPException(status_code=400, detail="method debe ser: collaborative, content_based, o hybrid")
    
    try:
        if method == "collaborative":
            recommendations = recommendation_system.get_cf_recommendations(user_id, n_recommendations)
        elif method == "content_based":
            # Para content-based, necesitamos películas que el usuario vio primero
            user_movies = recommendation_system.ratings_df[
                recommendation_system.ratings_df['user_id'] == user_id
            ]['movie_id'].values
            
            if len(user_movies) == 0:
                raise HTTPException(status_code=404, detail=f"Usuario {user_id} no tiene películas vistas")
            
            # Recomendar basado en la primera película que vio
            recommendations = recommendation_system.get_content_based_recommendations(
                user_movies[0], n_recommendations
            )
        else:  # hybrid
            recommendations = recommendation_system.get_hybrid_recommendations(
                user_id, n_recommendations
            )
        
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"No se encontraron recomendaciones para usuario {user_id}")
        
        from datetime import datetime
        
        return {
            "user_id": user_id,
            "method": method,
            "count": len(recommendations),
            "recommendations": [
                RecommendationResponse(
                    movie_id=movie_id,
                    title=title,
                    score=score
                )
                for movie_id, title, score in recommendations
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movies/{movie_id}/similar")
async def get_similar_movies(movie_id: int, n_similar: int = 5):
    """
    Obtiene películas similares usando Content-Based Filtering
    
    - **movie_id**: ID de la película
    - **n_similar**: Número de películas similares a retornar
    """
    
    if n_similar < 1 or n_similar > 20:
        raise HTTPException(status_code=400, detail="n_similar debe estar entre 1 y 20")
    
    try:
        recommendations = recommendation_system.get_content_based_recommendations(
            movie_id, n_similar
        )
        
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"Película {movie_id} no encontrada")
        
        return {
            "movie_id": movie_id,
            "similar_count": len(recommendations),
            "similar_movies": [
                MovieSimilarityResponse(
                    movie_id=mid,
                    title=title,
                    similarity_score=score
                )
                for mid, title, score in recommendations
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_system_stats():
    """Obtiene estadísticas del sistema de recomendación"""
    return {
        "total_users": recommendation_system.ratings_df['user_id'].nunique(),
        "total_movies": len(recommendation_system.movies_df),
        "total_ratings": len(recommendation_system.ratings_df),
        "avg_rating": round(recommendation_system.ratings_df['rating'].mean(), 2),
        "sparsity": round(
            (1 - len(recommendation_system.ratings_df) / 
             (recommendation_system.ratings_df['user_id'].nunique() * 
              len(recommendation_system.movies_df))) * 100, 2
        )
    }


@app.post("/recommendations")
async def get_custom_recommendations(
    user_id: int,
    method: str = "hybrid",
    n_recommendations: int = 5,
    cf_weight: Optional[float] = 0.7
):
    """
    Endpoint POST para obtener recomendaciones con parámetros personalizados
    
    - **user_id**: ID del usuario
    - **method**: Tipo de filtrado
    - **n_recommendations**: Número de recomendaciones
    - **cf_weight**: Peso de Collaborative Filtering (solo para hybrid)
    """
    
    try:
        if method == "hybrid" and cf_weight is not None:
            recommendations = recommendation_system.get_hybrid_recommendations(
                user_id, n_recommendations, cf_weight
            )
        else:
            # Usar endpoint GET
            pass
        
        from datetime import datetime
        
        return {
            "user_id": user_id,
            "method": method,
            "parameters": {
                "n_recommendations": n_recommendations,
                "cf_weight": cf_weight
            },
            "recommendations": [
                RecommendationResponse(
                    movie_id=movie_id,
                    title=title,
                    score=score
                )
                for movie_id, title, score in recommendations
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Iniciando servidor...")
    print("📚 Documentación API: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
