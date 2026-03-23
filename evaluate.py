"""
Script de evaluación del sistema de recomendación
Calcula métricas profesionales: RMSE, MAE, Precision@k, Recall@k
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.recommendation_system import MovieRecommendationSystem
import sys


def evaluate_model(ratings_df, user_movie_matrix, k=5):
    """
    Evalúa el modelo usando métricas estándar de recomendación
    
    Métricas:
    - RMSE (Root Mean Squared Error): Error en predicción de ratings
    - MAE (Mean Absolute Error): Error absoluto promedio
    - Precision@k: % de recomendaciones relevantes en top-k
    - Recall@k: % de items relevantes recuperados en top-k
    """
    
    print("\n" + "="*60)
    print("📊 EVALUACIÓN DEL MODELO")
    print("="*60)
    
    # Dividir datos: 80% entrenamiento, 20% test
    train_ratings, test_ratings = train_test_split(
        ratings_df, test_size=0.2, random_state=42
    )
    
    print(f"\n✓ Split: {len(train_ratings)} ratings en train, {len(test_ratings)} en test")
    
    # Entrenar modelo solo con datos de train
    print("\n🔄 Entrenando modelo con datos de train...")
    train_system = MovieRecommendationSystem()
    train_system.ratings_df = train_ratings
    train_system.movies_df = user_movie_matrix  # Hack para no cargar de disco
    train_system.load_data = lambda: None  # Skip load_data
    
    # Recrear matrices con datos de train
    train_system.user_movie_matrix = train_ratings.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating'
    ).fillna(0)
    
    # Recalcular similaridades solo con train
    from sklearn.metrics.pairwise import cosine_similarity
    train_system.user_similarity_matrix = pd.DataFrame(
        cosine_similarity(train_system.user_movie_matrix),
        index=train_system.user_movie_matrix.index,
        columns=train_system.user_movie_matrix.index
    )
    
    # Calcular RMSE y MAE
    print("\n📈 Calculando RMSE y MAE (vectorizado)...")
    
    # Pre-calcular predicciones para todos los pares user-movie en test
    test_pairs = test_ratings[['user_id', 'movie_id', 'rating']].copy()
    test_pairs['predicted'] = np.nan
    
    for user_id in test_pairs['user_id'].unique():
        if user_id not in train_system.user_similarity_matrix.index:
            continue
            
        similar_users = train_system.user_similarity_matrix[user_id].sort_values(ascending=False)[1:11]
        user_ratings = train_system.user_movie_matrix.loc[user_id]
        user_mask = test_pairs['user_id'] == user_id
        user_test_movies = test_pairs.loc[user_mask, 'movie_id'].values
        
        for movie_id in user_test_movies:
            if movie_id not in train_system.user_movie_matrix.columns:
                continue
            similar_users_who_watched = []
            for similar_user_id in similar_users.index:
                rating = train_system.user_movie_matrix.loc[similar_user_id, movie_id]
                if rating > 0:
                    similar_users_who_watched.append((similar_users[similar_user_id], rating))
            
            if similar_users_who_watched:
                total_sim = sum(s for s, _ in similar_users_who_watched)
                weighted = sum(s * r for s, r in similar_users_who_watched) / total_sim
                test_pairs.loc[(test_pairs['user_id'] == user_id) & (test_pairs['movie_id'] == movie_id), 'predicted'] = weighted
    
    valid_pairs = test_pairs.dropna(subset=['predicted'])
    predictions = valid_pairs['predicted'].values
    actuals = valid_pairs['rating'].values
    
    if len(predictions) > 0:
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        
        print(f"  RMSE: {rmse:.4f} (menor es mejor)")
        print(f"  MAE:  {mae:.4f} (menor es mejor)")
    else:
        print("  ⚠️  No se pudieron calcular RMSE/MAE")
    
    # Calcular Precision@k y Recall@k
    print(f"\n📌 Calculando Precision@{k} y Recall@{k}...")
    
    precisions = []
    recalls = []
    
    for user_id in test_ratings['user_id'].unique()[:100]:  # Muestra de usuarios
        # Películas del usuario en test que le gustaron (rating >= 4)
        user_test_movies = test_ratings[
            (test_ratings['user_id'] == user_id) & 
            (test_ratings['rating'] >= 4)
        ]['movie_id'].values
        
        if len(user_test_movies) == 0:
            continue
        
        # Recomendaciones del modelo
        try:
            recs = train_system.get_cf_recommendations(user_id, n_recommendations=k)
            rec_movies = [mid for _, mid, _ in recs]
            
            # Calcular precision y recall
            relevant = len(set(user_test_movies) & set(rec_movies))
            
            precision = relevant / k if k > 0 else 0
            recall = relevant / len(user_test_movies) if len(user_test_movies) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        except:
            pass
    
    if precisions:
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        
        print(f"  Precision@{k}: {avg_precision:.4f} (promedio de {len(precisions)} usuarios)")
        print(f"  Recall@{k}:    {avg_recall:.4f}")
    else:
        print("  ⚠️  No se pudieron calcular Precision/Recall")
    
    # Coverage (% de películas que pueden ser recomendadas)
    total_movies = ratings_df['movie_id'].nunique()
    print(f"\n📺 Coverage: {(len(ratings_df['movie_id'].unique()) / total_movies * 100):.1f}%")
    print(f"   ({len(ratings_df['movie_id'].unique())} de {total_movies} películas)")
    
    print("\n" + "="*60)


def main():
    # Crear y entrenar modelo
    print("🚀 Iniciando entrenamiento del modelo...\n")
    
    system = MovieRecommendationSystem()
    system.train()
    
    # Evaluación
    evaluate_model(system.ratings_df, system.movies_df)
    
    # Ejemplo de predicción
    print("\n" + "="*60)
    print("🎬 EJEMPLO DE RECOMENDACIONES")
    print("="*60)
    
    test_user_id = 1
    print(f"\nRecomendaciones Collaborative Filtering para Usuario {test_user_id}:")
    cf_recs = system.get_cf_recommendations(test_user_id, n_recommendations=5)
    for i, (movie_id, title, score) in enumerate(cf_recs, 1):
        print(f"  {i}. {title[:50]:50} (score: {score})")
    
    print(f"\nRecomendaciones Híbridas para Usuario {test_user_id}:")
    hybrid_recs = system.get_hybrid_recommendations(test_user_id, n_recommendations=5)
    for i, (movie_id, title, score) in enumerate(hybrid_recs, 1):
        print(f"  {i}. {title[:50]:50} (score: {score})")
    
    # Content-based para una película
    test_movie_id = 1
    print(f"\nPelículas similares a '{system.movies_df[system.movies_df['movie_id']==test_movie_id]['title'].values[0]}':")
    cb_recs = system.get_content_based_recommendations(test_movie_id, n_recommendations=5)
    for i, (movie_id, title, score) in enumerate(cb_recs, 1):
        print(f"  {i}. {title[:50]:50} (similarity: {score})")
    
    print("\n✅ Evaluación completada\n")


if __name__ == '__main__':
    main()
