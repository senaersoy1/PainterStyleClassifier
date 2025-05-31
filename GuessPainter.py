import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import glob
import matplotlib.pyplot as plt

def test_external_images(test_folder, pca, knn, label_encoder):
    test_images = sorted(glob.glob(os.path.join(test_folder, "bouguereau_image*.jpg")))

    artist_totals = {
        "Alexandre Cabanel": 0.0,
        "Caravaggio": 0.0,
        "Monet": 0.0
    }
    total_images = 0

    print("\n--- Bouguereau Görselleri Tahminleri ---")
    for path in test_images:
        image = Image.open(path).convert("RGB").resize((64, 64))
        x = np.asarray(image).flatten()
        result = predict_image_vector(x, pca, knn, label_encoder)
        print(f"\nGorsel: {os.path.basename(path)}")
        for artist, score in result.items():
            print(f"{artist}: {score}%")
            if artist in artist_totals:
                artist_totals[artist] += score
        total_images += 1

    # Ortalama benzerlik hesapla
    if total_images > 0:
        for artist in artist_totals:
            artist_totals[artist] /= total_images

        # --- Grafik ---
        plt.figure(figsize=(6, 5))
        plt.bar(artist_totals.keys(), artist_totals.values(), color=["skyblue", "lightcoral", "gold"])
        plt.title("Bouguereau Tablolarinin Ortalama Benzerligi")
        plt.ylabel("Ortalama Benzerlik (%)")
        plt.ylim(0, 100)
        for i, val in enumerate(artist_totals.values()):
            plt.text(i, val + 2, f"{val:.2f}%", ha='center', fontsize=10)
        plt.tight_layout()
        plt.show()


# --- 1. Görsel yükleyici ---
def load_images(folder_path, image_size=(64, 64)):
    X = []
    y = []
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                if filename.lower().endswith('.jpg'):
                    path = os.path.join(label_folder, filename)
                    image = Image.open(path).convert("RGB").resize(image_size)
                    X.append(np.asarray(image).flatten())
                    y.append(label)
    return np.array(X), np.array(y)

# --- 2. Yüzdelik tahmin fonksiyonu ---
def predict_image_vector(x, pca, knn, label_encoder):
    x_pca = pca.transform(x.reshape(1, -1))
    distances, indices = knn.kneighbors(x_pca, n_neighbors=5)
    neighbor_labels = knn._y[indices[0]]
    labels, counts = np.unique(neighbor_labels, return_counts=True)
    result = {}
    for label, count in zip(labels, counts):
        artist = label_encoder.inverse_transform([label])[0]
        result[artist] = round((count / 5) * 100, 2)
    return result

# --- 3. Ana çalışma ---
if __name__ == "__main__":
    dataset_path = r"C:\Users\HP\Desktop\MYZ307-Project"

    print("Veri yukleniyor...")
    X, y = load_images(dataset_path)
    print("Toplam gorsel sayisi:", len(X))
    print("Sinif dagilimi:", Counter(y))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Eğitim/test ayırımı
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    print(f"Egitim seti: {len(X_train)} gorsel")
    print(f"Test seti: {len(X_test)} gorsel")

    # PCA ve KNN
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train_pca, y_train)
    
    # Test seti üzerinde tahminler
    print("\n--- Test Gorselleri Tahminleri ---")
    for i in range(len(X_test)):
        result = predict_image_vector(X_test[i], pca, knn, label_encoder)
        true_label = label_encoder.inverse_transform([y_test[i]])[0]
        print(f"\nGercek: {true_label}")
        for artist, score in result.items():
            print(f"{artist}: {score}%")
    
    test_external_images(".", pca, knn, label_encoder)  # Aynı klasörde olduğu için "."


