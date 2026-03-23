import os
import urllib.request
import zipfile
import pandas as pd

def download_movielens_data():
    """Descarga y extrae el dataset MovieLens 100K"""
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Verificar si ya existe
    if os.path.exists(os.path.join(data_dir, 'u.data')):
        print("✓ Dataset ya existe")
        return
    
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    zip_path = os.path.join(data_dir, 'ml-100k.zip')
    
    print("📥 Descargando MovieLens 100K dataset...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("📦 Extrayendo archivos...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Mover archivos a la carpeta data
    import shutil
    src = os.path.join(data_dir, 'ml-100k')
    for file in os.listdir(src):
        shutil.move(os.path.join(src, file), os.path.join(data_dir, file))
    os.rmdir(src)
    os.remove(zip_path)
    
    print("✓ Dataset descargado exitosamente")
    print(f"✓ Ubicación: {data_dir}/")

if __name__ == '__main__':
    download_movielens_data()