# Müşteri Segmentasyonu Projesi
# AVM müşterilerini gelir ve harcama skoruna göre gruplandırma

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Veriyi yükle
df = pd.read_csv('Mall_Customers.csv')
print("Data set installed")
print(df.head())
print(f"\nToplam {len(df)} müşteri var")

# Temel bilgiler
print("\nData özeti:")
print(df.describe())

print("\nEksik value var mı?")
print(df.isnull().sum())

# Sütun isimlerini değiştir
df.columns = ['MusteriID', 'Cinsiyet', 'Yas', 'YillikGelir', 'HarcamaSkoru']

# Görselleştirme
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0,0].hist(df['Yas'], bins=15, color='skyblue', edgecolor='black')
ax[0,0].set_title('Yas Dagilimi')

ax[0,1].hist(df['YillikGelir'], bins=15, color='lightgreen', edgecolor='black')
ax[0,1].set_title('Yillik Gelir Dagilimi')

ax[1,0].hist(df['HarcamaSkoru'], bins=15, color='salmon', edgecolor='black')
ax[1,0].set_title('Harcama Skoru Dagilimi')

ax[1,1].pie(df['Cinsiyet'].value_counts(), labels=['Kadin','Erkek'], autopct='%1.1f%%')
ax[1,1].set_title('Cinsiyet Dagilimi')

plt.tight_layout()
plt.savefig('1_veri_dagilimi.png')
plt.close()

# Gelir vs Harcama grafiği
plt.figure(figsize=(8,6))
plt.scatter(df['YillikGelir'], df['HarcamaSkoru'], alpha=0.6)
plt.xlabel('Yillik Gelir (bin $)')
plt.ylabel('Harcama Skoru')
plt.title('Gelir - Harcama Iliskisi')
plt.savefig('2_gelir_harcama.png')
plt.close()

# Kümeleme için veri hazırlığı
X = df[['YillikGelir', 'HarcamaSkoru']].values

# Ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optimal küme sayısını bul - Elbow yöntemi
wcss = []
sil_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Elbow ve Silhouette grafikleri
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(range(2,11), wcss, 'o-')
ax[0].set_xlabel('Kume Sayisi')
ax[0].set_ylabel('WCSS')
ax[0].set_title('Elbow Yontemi')
ax[0].axvline(x=5, color='r', linestyle='--')

ax[1].plot(range(2,11), sil_scores, 'o-', color='green')
ax[1].set_xlabel('Kume Sayisi')
ax[1].set_ylabel('Silhouette Skoru')
ax[1].set_title('Silhouette Analizi')
ax[1].axvline(x=5, color='r', linestyle='--')

plt.tight_layout()
plt.savefig('3_optimal_k.png')
plt.close()

print("\nSilhouette skorları:")
for i, s in enumerate(sil_scores):
    print(f"k={i+2}: {s:.3f}")

# k=5 ile model oluştur
k = 5
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
df['Kume'] = kmeans.fit_predict(X_scaled)

print(f"\nModel k={k} ile eğitildi")
print(f"Silhouette skoru: {silhouette_score(X_scaled, df['Kume']):.3f}")

# Kümeleri görselleştir
plt.figure(figsize=(10,7))
colors = ['red', 'blue', 'green', 'orange', 'purple']

for i in range(k):
    cluster = df[df['Kume'] == i]
    plt.scatter(cluster['YillikGelir'], cluster['HarcamaSkoru'],
                c=colors[i], label=f'Kume {i}', s=60, alpha=0.7)

# Küme merkezleri
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], c='black', marker='X', s=200, label='Merkezler')

plt.xlabel('Yillik Gelir (bin $)')
plt.ylabel('Harcama Skoru')
plt.title('Musteri Segmentasyonu')
plt.legend()
plt.savefig('4_kumeleme.png')
plt.close()

# Küme analizi
print("\nKüme Analizi")
for i in range(k):
    kume = df[df['Kume'] == i]
    print(f"\nKüme {i}:")
    print(f"Müşteri sayısı: {len(kume)}")
    print(f"Ort. gelir: {kume['YillikGelir'].mean():.1f}k$")
    print(f"Ort. harcama: {kume['HarcamaSkoru'].mean():.1f}")
    print(f"Ort. yaş: {kume['Yas'].mean():.1f}")

# Sonucu kaydet
df.to_csv('sonuc.csv', index=False)
print("\nSonuçlar sonuc.csv dosyasına kaydedildi")
