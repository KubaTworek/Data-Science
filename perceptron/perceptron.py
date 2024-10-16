# Importowanie potrzebnych bibliotek
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Definicja klasy Perceptron (ręczna implementacja)
class PerceptronManual:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Inicjalizacja perceptronu.
        
        :param learning_rate: współczynnik uczenia, kontroluje tempo dostosowania wag
        :param n_iters: liczba iteracji (epok) dla procesu uczenia
        """
        self.lr = learning_rate  # współczynnik uczenia
        self.n_iters = n_iters   # liczba epok (ile razy algorytm przejdzie przez dane)
        self.activation_func = self._unit_step_function  # funkcja aktywacji - tutaj funkcja progowa
        self.weights = None  # wagi (ustawiane w funkcji fit)
        self.bias = None  # bias (inicjowane w funkcji fit)

    # Funkcja aktywacji (unit step function), która zwraca 1, jeśli suma wagowa >= 0, inaczej 0
    def _unit_step_function(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """
        Trenuje perceptron na danych X (cechy) i y (etykiety).
        
        :param X: macierz cech, kształt (n_samples, n_features)
        :param y: wektor etykiet, kształt (n_samples,)
        """
        n_samples, n_features = X.shape  # n_samples - liczba próbek, n_features - liczba cech

        # Inicjalizacja wag na wartości zerowe oraz biasu na 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Iteracje (epoki) procesu uczenia perceptronu
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Obliczenie sumy wagowej
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Przepuszczenie wartości przez funkcję aktywacji
                y_predicted = self.activation_func(linear_output)

                # Obliczenie błędu (y rzeczywiste - y przewidywane)
                update = self.lr * (y[idx] - y_predicted)
                # Aktualizacja wag i biasu na podstawie błędu
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Dokonuje predykcji na nowych danych X.
        
        :param X: macierz cech, kształt (n_samples, n_features)
        :return: wektor predykcji (0 lub 1)
        """
        # Obliczenie sumy wagowej
        linear_output = np.dot(X, self.weights) + self.bias
        # Zastosowanie funkcji aktywacji do wyniku liniowego
        y_predicted = self.activation_func(linear_output)
        return y_predicted


# Wczytywanie danych z pliku CSV do obiektu DataFrame
file_path = r"ideaprojects\data-science\perceptron\wdbc.data"
columns = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", 
           "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", 
           "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", 
           "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", 
           "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", 
           "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]

# Wczytanie danych do obiektu DataFrame
diag = pd.read_csv(file_path, names=columns)

# Usunięcie kolumny 'id', która nie wnosi istotnych informacji (np. nie przyczynia się do predykcji)
diag = diag.drop(columns=['id'])

# Sprawdzanie, jakie dane mamy przed konwersją do typów liczbowych
print("\nPrzykładowe wartości w kolumnach przed konwersją:")
print(diag.head())

# Konwersja wszystkich kolumn oprócz 'diagnosis' na wartości numeryczne (float).
# Jeśli znajdziemy błędne wartości (np. tekst), zamieniamy je na NaN.
for col in diag.columns:
    if col != 'diagnosis':  # Pomijamy kolumnę diagnosis
        diag[col] = pd.to_numeric(diag[col], errors='coerce')

# Sprawdzenie liczby braków danych po konwersji do wartości liczbowych
print("\nLiczba brakujących wartości (NaN) w każdej kolumnie po konwersji:")
print(diag.isnull().sum())

# Usunięcie wierszy z brakującymi danymi (opcjonalnie)
diag = diag.dropna()

# Zamiana wartości w kolumnie 'diagnosis' na numeryczne: 'M' (złośliwy) -> 1, 'B' (łagodny) -> 0
diag['diagnosis'] = diag['diagnosis'].apply(lambda d: 1 if d == 'M' else 0)

# ----- 1. Sprawdzanie braków danych -----
print("\nBraki danych w każdej kolumnie:")
print(diag.isnull().sum())


# ----- Eksploracyjna analiza danych (EDA) -----

# ----- 1. Rozkład zmiennej docelowej 'diagnosis' -----
sns.countplot(x='diagnosis', data=diag)
plt.title('Rozkład zmiennej docelowej (diagnosis)')
plt.show()

# ----- 2. Sprawdzanie wariancji cech -----
variances = diag.var()
print("\nWariancja cech:")
print(variances)

# ----- 3. Macierz korelacji -----
plt.figure(figsize=(16, 10))
corr_matrix = diag.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Macierz korelacji")
plt.show()

# Wyświetlenie korelacji każdej cechy z diagnozą (diagnosis)
correlation_with_diagnosis = corr_matrix['diagnosis'].sort_values(ascending=False)
print("\nKorelacja cech z diagnozą (diagnosis):")
print(correlation_with_diagnosis)

# ----- 4. Wybór najważniejszych cech -----
significant_features = corr_matrix['diagnosis'][corr_matrix['diagnosis'].abs() > 0.75].index.tolist()
significant_features.remove('diagnosis')
print(f"Najbardziej istotne cechy: {significant_features}")

# ----- 5. Analiza rozkładu zmiennych -----
diag[significant_features].hist(bins=30, figsize=(10, 6))
plt.suptitle('Histogramy dla wybranych cech')
plt.show()

# ----- 6. Wykresy parowe (Pairplot) -----
sns.pairplot(diag[significant_features + ['diagnosis']], hue='diagnosis')
plt.title("Pairplot dla wybranych cech i diagnozy")
plt.show()

# ----- 7. Transformacje cech -----
diag[significant_features] = np.log1p(diag[significant_features])  # logarytm naturalny z istotnych cech
sns.histplot(diag[significant_features], kde=True)
plt.title("Logarytmiczna transformacja istotnych cech")
plt.show()

# ----- 8. Wizualizacja danych przed i po standaryzacji -----
# Boxplot przed standaryzacją
plt.figure(figsize=(10, 6))
sns.boxplot(data=diag[significant_features])
plt.title('Boxplot dla istotnych cech przed standaryzacją')
plt.show()

# Standaryzacja zmiennych
scaler = StandardScaler()
X_std = scaler.fit_transform(diag[significant_features])

# Boxplot po standaryzacji
plt.figure(figsize=(10, 6))
sns.boxplot(data=X_std)
plt.title('Boxplot dla istotnych cech po standaryzacji')
plt.show()

# ----- 9. Usuwanie wartości odstających (Outliers) za pomocą IQR -----
Q1 = diag.quantile(0.25)
Q3 = diag.quantile(0.75)
IQR = Q3 - Q1

# Definiowanie wartości odstających
outliers = ((diag < (Q1 - 1.5 * IQR)) | (diag > (Q3 + 1.5 * IQR)))
print("\nLiczba wartości odstających w każdej kolumnie:")
print(outliers.sum())

# Usuwanie odchyłów
diag = diag[~outliers.any(axis=1)]
print(f"Liczba próbek po usunięciu odchyłów: {len(diag)}")

# ----- 10. Analiza składowych głównych (PCA) -----
X = diag[significant_features]
y = diag['diagnosis']

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

# ---- 10. Trenowanie perceptronu ---- #

# Wybieramy tylko najbardziej istotne cechy do dalszej analizy
X = diag[significant_features]
y = diag['diagnosis']

# Normalizacja cech (standaryzacja)
X_std = scaler.fit_transform(X)

# Podział danych na zbiór treningowy (80%) i testowy (20%)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# Inicjalizacja perceptronu z parametrami: współczynnik uczenia 0.01 i maksymalna liczba iteracji 100
perceptron = Perceptron(eta0=0.01, max_iter=100)

# Trenowanie perceptronu na zbiorze treningowym
perceptron.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = perceptron.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)  # Obliczenie dokładności modelu
print(f"\nDokładność perceptronu: {accuracy * 100:.2f}%")

# Wyświetlenie raportu klasyfikacji (precision, recall, f1-score)
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))

# Tworzenie macierzy pomyłek (confusion matrix) i jej wizualizacja
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Macierz Pomyłek - Perceptron")
plt.xlabel("Przewidywane etykiety")
plt.ylabel("Rzeczywiste etykiety")
plt.show()
