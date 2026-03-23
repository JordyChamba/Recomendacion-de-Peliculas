"""
Sistema de Recomendación de Películas
Implementa: Collaborative Filtering + Content-Based Filtering
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class MovieRecommendationSystem:
    """
    Sistema de recomendación de películas con dos enfoques:
    1. Collaborative Filtering (basado en similitud entre usuarios)
    2. Content-Based (basado en características de películas)
    """
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.ratings_df = None
        self.movies_df = None
        self.user_movie_matrix = None
        self.user_similarity_matrix = None
        self.movie_similarity_matrix = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Carga los datos de MovieLens 100K"""
        print("📂 Cargando datos...")
        
        # Cargar ratings
        self.ratings_df = pd.read_csv(
            os.path.join(self.data_dir, 'u.data'),
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        
        # Cargar información de películas
        self.movies_df = pd.read_csv(
            os.path.join(self.data_dir, 'u.item'),
            sep='|',
            names=['movie_id', 'title', 'release_date', 'video_release_date',
                   'imdb_url', 'unknown', 'action', 'adventure', 'animation',
                   'childrens', 'comedy', 'crime', 'documentary', 'drama',
                   'fantasy', 'film_noir', 'horror', 'musical', 'mystery',
                   'romance', 'sci_fi', 'thriller', 'war', 'western'],
            encoding='latin-1'
        )
        
        print(f"✓ {len(self.ratings_df)} ratings cargados")
        print(f"✓ {len(self.movies_df)} películas cargadas")
        print(f"✓ {self.ratings_df['user_id'].nunique()} usuarios únicos")
        
        return self
    
    def preprocess_data(self):
        """Preprocesa y limpia los datos"""
        print("\n🔧 Preprocesando datos...")
        
        # Estadísticas básicas
        print(f"  Rating promedio: {self.ratings_df['rating'].mean():.2f}")
        print(f"  Rango de ratings: {self.ratings_df['rating'].min()}-{self.ratings_df['rating'].max()}")
        
        # Crear matriz usuario-película
        self.user_movie_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating'
        )
        
        # Rellenar NaNs con 0 (películas no vistas)
        self.user_movie_matrix = self.user_movie_matrix.fillna(0)
        
        print(f"✓ Matriz usuario-película creada: {self.user_movie_matrix.shape}")
        
        return self
    
    def build_collaborative_filtering(self):
        """Construye la matriz de similitud entre usuarios (Collaborative Filtering)"""
        print("\n🤝 Construyendo modelo Collaborative Filtering...")
        
        # Calcular similitud coseno entre usuarios
        self.user_similarity_matrix = cosine_similarity(self.user_movie_matrix)
        self.user_similarity_matrix = pd.DataFrame(
            self.user_similarity_matrix,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )
        
        print(f"✓ Matriz de similitud de usuarios creada: {self.user_similarity_matrix.shape}")
        
        return self
    
    def build_content_based(self):
        """Construye la matriz de similitud entre películas (Content-Based)"""
        print("\n📽️  Construyendo modelo Content-Based...")
        
        # Extraer características de géneros (últimas 19 columnas son géneros binarios)
        genres = self.movies_df.iloc[:, 6:].values
        
        # Calcular similitud coseno entre películas
        self.movie_similarity_matrix = cosine_similarity(genres)
        self.movie_similarity_matrix = pd.DataFrame(
            self.movie_similarity_matrix,
            index=self.movies_df['movie_id'],
            columns=self.movies_df['movie_id']
        )
        
        print(f"✓ Matriz de similitud de películas creada: {self.movie_similarity_matrix.shape}")
        
        return self
    
    def get_cf_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Tuple]:
        """
        Genera recomendaciones usando Collaborative Filtering
        
        Algoritmo:
        1. Encuentra usuarios similares al user_id
        2. Busca películas que ellos vieron y el usuario no
        3. Predice rating basado en la similitud
        """
        if user_id not in self.user_similarity_matrix.index:
            print(f"⚠️  Usuario {user_id} no encontrado")
            return []
        
        # Obtener similitud con otros usuarios (excluir el usuario mismo)
        similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False)[1:11]
        
        # Películas que el usuario ha visto
        user_watched = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id'])
        
        # Películas no vistas
        all_movies = set(self.movies_df['movie_id'])
        unwatched_movies = all_movies - user_watched
        
        # Predicciones de rating
        recommendations = {}
        for movie_id in unwatched_movies:
            # Encuentra usuarios similares que vieron esta película
            similar_users_who_watched = []
            for similar_user_id in similar_users.index:
                if self.user_movie_matrix.loc[similar_user_id, movie_id] > 0:
                    rating = self.user_movie_matrix.loc[similar_user_id, movie_id]
                    similarity = similar_users[similar_user_id]
                    similar_users_who_watched.append((similarity, rating))
            
            if similar_users_who_watched:
                # Promedio ponderado de ratings
                total_similarity = sum(s for s, _ in similar_users_who_watched)
                weighted_rating = sum(s * r for s, r in similar_users_who_watched) / total_similarity
                recommendations[movie_id] = weighted_rating
        
        # Ordenar y retornar top N
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        return self._format_recommendations(top_recommendations)
    
    def get_content_based_recommendations(self, movie_id: int, n_recommendations: int = 5) -> List[Tuple]:
        """
        Genera recomendaciones usando Content-Based Filtering
        
        Algoritmo:
        1. Encuentra películas similares en géneros
        2. Ordena por similitud
        """
        if movie_id not in self.movie_similarity_matrix.index:
            print(f"⚠️  Película {movie_id} no encontrada")
            return []
        
        # Obtener películas similares (excluir la película misma)
        similar_movies = self.movie_similarity_matrix[movie_id].sort_values(ascending=False)[1:n_recommendations+1]
        
        return self._format_recommendations(
            [(movie_id, score) for movie_id, score in similar_movies.items()]
        )
    
    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5, 
                                   cf_weight: float = 0.7) -> List[Tuple]:
        """
        Genera recomendaciones usando enfoque híbrido
        Combina Collaborative Filtering + Content-Based
        """
        cf_recs = self.get_cf_recommendations(user_id, n_recommendations * 2)
        
        if not cf_recs:
            return []
        
        # Para cada película en CF, buscar similares en CB
        all_recs = {}
        for movie_id, _, cf_score in cf_recs:
            all_recs[movie_id] = cf_score * cf_weight
            
            # Agregar películas similares con peso menor
            cb_recs = self.get_content_based_recommendations(movie_id, 3)
            for similar_id, _, cb_score in cb_recs:
                if similar_id not in all_recs:
                    all_recs[similar_id] = cb_score * (1 - cf_weight)
        
        # Ordenar y retornar top N
        top_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        return self._format_recommendations(top_recs)
    
    def _format_recommendations(self, recommendations: List[Tuple]) -> List[Tuple]:
        """Formatea recomendaciones con títulos de películas"""
        formatted = []
        for movie_id, score in recommendations:
            movie_title = self.movies_df[self.movies_df['movie_id'] == movie_id]['title'].values
            if len(movie_title) > 0:
                formatted.append((movie_id, movie_title[0], round(score, 3)))
        
        return formatted
    
    def save_model(self, model_path: str = 'models/recommendation_model.pkl'):
        """Guarda el modelo entrenado"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'user_movie_matrix': self.user_movie_matrix,
            'user_similarity_matrix': self.user_similarity_matrix,
            'movie_similarity_matrix': self.movie_similarity_matrix,
            'movies_df': self.movies_df,
            'ratings_df': self.ratings_df
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Modelo guardado en {model_path}")
    
    def load_model(self, model_path: str = 'models/recommendation_model.pkl'):
        """Carga un modelo previamente entrenado"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_movie_matrix = model_data['user_movie_matrix']
        self.user_similarity_matrix = model_data['user_similarity_matrix']
        self.movie_similarity_matrix = model_data['movie_similarity_matrix']
        self.movies_df = model_data['movies_df']
        self.ratings_df = model_data['ratings_df']
        
        print(f"✓ Modelo cargado desde {model_path}")
    
    def train(self):
        """Pipeline completo de entrenamiento"""
        self.load_data()
        self.preprocess_data()
        self.build_collaborative_filtering()
        self.build_content_based()
        self.save_model()
        print("\n✅ Entrenamiento completado")
        return self
