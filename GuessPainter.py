import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import glob
import matplotlib.pyplot as plt

def test_external_images(test_folder, image_pattern, pca, knn, label_encoder, artist_names, title):
    """
    test_folder: Test resimlerinin olduğu klasör yolu
    image_pattern: Resim dosyalarının isim kalıbı, örn: "bouguereau_image*.jpg"
    artist_names: Benzerlik skorlarının toplanacağı sanatçı isimleri listesi
    title: Grafik başlığı
    """
    test_images = sorted(glob.glob(os.path.join(test_folder, image_pattern)))

    artist_totals = {artist: 0.0 for artist in artist_names}
    total_images = 0

    print(f"\n--- {title} Görselleri Tahminleri ---")
    for path in test_images:
        image = Image.open(path).convert("RGB").resize((64, 64))
        x = np.asarray(image).flatten()
        result = predict_image_vector(x, pca, knn, label_encoder)
        print(f"\nGörsel: {os.path.basename(path)}")
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
        plt.bar(artist_totals.keys(), artist_totals.values())
        plt.title(title)
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

    # Test seti tahminleri için en yüksek benzerlik yüzdesi olan etiketi al
    y_pred = []
    for i in range(len(X_test)):
        result = predict_image_vector(X_test[i], pca, knn, label_encoder)
        # En yüksek skoru veren sanatçıyı bul
        pred_artist = max(result, key=result.get)
        y_pred.append(pred_artist)

    # Gerçek etiketleri string haline çevir
    y_true = label_encoder.inverse_transform(y_test)

    # Doğruluk hesapla
    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Seti Doğruluğu: {acc*100:.2f}%")

    # Karışıklık matrisi hesapla
    cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)

    # Karışıklık matrisini görselleştir
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Karışıklık Matrisi")
    plt.tight_layout()
    plt.show()


    # Bouguereau için test
    test_external_images(
        test_folder=".",
        image_pattern="bouguereau_image*.jpg",
        pca=pca,
        knn=knn,
        label_encoder=label_encoder,
        artist_names=["Alexandre Cabanel", "Caravaggio", "Monet"],
        title="Bouguereau Tablolarinin Ortalama Benzerligi"
    )

    # Artemisia Gentileschi için test
    test_external_images(
        test_folder=".",
        image_pattern="artemisia_image*.jpg",
        pca=pca,
        knn=knn,
        label_encoder=label_encoder,
        artist_names=["Alexandre Cabanel", "Caravaggio", "Monet"],
        title="Artemisia Gentileschi Tablolarinin Ortalama Benzerligi"
    )

    # Pierre-Auguste Renoir için test
    test_external_images(
        test_folder=".",
        image_pattern="renoir_image*.jpg",
        pca=pca,
        knn=knn,
        label_encoder=label_encoder,
        artist_names=["Alexandre Cabanel", "Caravaggio", "Monet"],
        title="Pierre-Auguste Renoir Tablolarinin Ortalama Benzerligi"
    )

    
