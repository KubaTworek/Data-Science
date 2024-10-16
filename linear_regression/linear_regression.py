import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA

# ----- 1. Wczytywanie danych -----
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv(r"ideaprojects\data-science\linear_regression\housing.data", delim_whitespace=True, header=None, names=cols)

# ----- 2. Sprawdzanie danych -----
print("Sprawdzenie typów danych i pierwszych wierszy danych:")
print(data.dtypes)
print(data.head())

# ----- 3. Konwersja danych -----
for col in cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# ----- 4. Sprawdzanie brakujących danych -----
print("\nBrakujące wartości w kolumnach po konwersji:")
print(data.isnull().sum())

# ----- 5. Usuwanie brakujących danych -----
data = data.dropna()

# ----- Eksploracyjna analiza danych (EDA) -----

# 1. Rozkład zmiennej zależnej (MEDV - cena domów)
plt.figure(figsize=(10, 6))
sns.histplot(data['MEDV'], kde=True, bins=30)
plt.title("Rozkład ceny domów (MEDV)")
plt.xlabel("Cena domów")
plt.ylabel("Liczba domów")
plt.show()

# 2. Sprawdzanie wariancji cech
variances = data.var()
print("\nWariancja cech:")
print(variances)

# 3. Macierz korelacji
plt.figure(figsize=(12, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Macierz korelacji")
plt.show()

# Wyświetlenie korelacji każdej cechy z ceną (MEDV)
correlation_with_price = corr_matrix['MEDV'].sort_values(ascending=False)
print("\nKorelacja cech z ceną (MEDV):")
print(correlation_with_price)

# 4. Wybór najważniejszych cech
significant_features = corr_matrix['MEDV'][corr_matrix['MEDV'].abs() > 0.5].index.tolist()
significant_features.remove('MEDV')
print(f"Najbardziej istotne cechy: {significant_features}")

# 5. Analiza rozkładu zmiennych
data[significant_features].hist(bins=30, figsize=(10, 6))
plt.suptitle('Histogramy dla wybranych cech')
plt.show()

# 6. Wykresy parowe (Pairplot)
sns.pairplot(data[significant_features + ['MEDV']])
plt.title("Pairplot dla zmiennych istotnych i ceny domów")
plt.show()

# 7. Transformacje cech
data[significant_features] = np.log1p(data[significant_features])  # logarytm naturalny z istotnych cech
plt.figure(figsize=(10, 6))
sns.histplot(data[significant_features], kde=True)
plt.title("Logarytmiczna transformacja istotnych cech")
plt.show()

# 8. Wizualizacja danych przed i po standaryzacji
# Boxplot przed standaryzacją
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[significant_features])
plt.title('Boxplot dla istotnych cech przed standaryzacją')
plt.show()

# Standaryzacja zmiennych
scaler = StandardScaler()
X_std = scaler.fit_transform(data[significant_features])

# Boxplot po standaryzacji
plt.figure(figsize=(10, 6))
sns.boxplot(data=X_std)
plt.title('Boxplot dla istotnych cech po standaryzacji')
plt.show()

# 9. Usuwanie wartości odstających (Outliers) za pomocą IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Definiowanie wartości odstających
outlier_condition = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
print("\nLiczba wartości odstających w każdej kolumnie:")
print(outlier_condition.sum())

# Usuwanie odchyłów
data = data[~outlier_condition.any(axis=1)]
print(f"Liczba próbek po usunięciu odchyłów: {len(data)}")

# 10. Analiza składowych głównych (PCA)
X = data[significant_features]
y = data['MEDV']

X_std = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

print(f"\nWyjaśniona wariancja przez komponenty PCA: {pca.explained_variance_ratio_}")

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm')
plt.title("Wyniki PCA (2 składowe)")
plt.xlabel('Pierwsza składowa')
plt.ylabel('Druga składowa')
plt.show()

# ----- Przygotowanie danych do modelowania -----
X = data[significant_features].values
y = data['MEDV'].values

# Standaryzacja cech
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Podział danych na zbiory treningowe (80%) i testowe (20%)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# ----- Dopasowanie modelu regresji liniowej -----
lr = LinearRegression()
lr.fit(X_train, y_train)

# ----- Ocena modelu -----
y_pred = lr.predict(X_test)

# ----- Wizualizacja wyników -----
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, label="Test Data")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color="red")
plt.xlabel("Prawdziwe wartości (y_test)")
plt.ylabel("Przewidywane wartości (y_pred)")
plt.title("Regresja liniowa - Zbiór testowy")
plt.legend()
plt.show()

# ----- Dodatkowa analiza skuteczności modelu -----
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMetryki dla zbioru testowego:\nMAE: {mae}\nMSE: {mse}\nR²: {r2}")

# Wykres residuów
residuals = y_test - y_pred

plt.figure(figsize=(7, 7))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Wykres residuów (różnice rzeczywiste - przewidywane)")
plt.xlabel("Przewidywane wartości")
plt.ylabel("Residuły")
plt.show()

# Histogram residuów
plt.figure(figsize=(7, 7))
sns.histplot(residuals, kde=True)
plt.title("Rozkład residuów (błędy)")
plt.xlabel("Residuły")
plt.ylabel("Częstotliwość")
plt.show()
