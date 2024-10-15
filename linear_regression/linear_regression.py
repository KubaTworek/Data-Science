import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----- 1. Wczytywanie danych -----
# Wczytujemy dane dotyczące cen domów, z różnych cech (takich jak wskaźnik przestępczości, liczba pokoi, itp.).
# Używamy delimitera opartego na białych znakach, ponieważ dane są rozdzielone spacjami.
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv(r"ideaprojects\data-science\linear_regression\housing.data", delim_whitespace=True, header=None, names=cols)

# ----- 2. Sprawdzanie danych -----
# Sprawdzamy typy danych, aby upewnić się, że wszystko zostało wczytane prawidłowo.
# Sprawdzamy również pierwsze kilka wierszy, aby zrozumieć strukturę danych.
print("Sprawdzenie typów danych i pierwszych wierszy danych:")
print(data.dtypes)
print(data.head())

# ----- 3. Konwersja danych -----
# Konwersja wszystkich kolumn na liczby. Jeśli znajdziemy jakiekolwiek wartości błędne (np. teksty w liczbowych kolumnach),
# konwertujemy je na NaN (wartości brakujące). To ważne, aby móc poprawnie analizować i modelować dane.
for col in cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# ----- 4. Sprawdzanie brakujących danych -----
# Sprawdzamy, ile brakujących wartości znajduje się w każdej kolumnie po konwersji.
# Ma to na celu upewnienie się, że dane są czyste i gotowe do analizy.
print("\nBrakujące wartości w kolumnach po konwersji:")
print(data.isnull().sum())

# ----- 5. Usuwanie brakujących danych -----
# Usuwamy wiersze, które zawierają brakujące dane, ponieważ nie mogą one być użyte w modelu.
# Może to obejmować kilka lub kilkadziesiąt wierszy, ale model będzie działał lepiej na pełnych danych.
data = data.dropna()

# ----- 6. Eksploracyjna analiza danych (EDA) -----
# Przeprowadzamy eksploracyjną analizę danych, aby lepiej zrozumieć ich strukturę i znaleźć kluczowe zależności.

# 1. Rozkład zmiennej zależnej (MEDV - cena domów)
# Tworzymy histogram, aby zobaczyć rozkład cen domów. Jest to istotne, aby zrozumieć, czy mamy do czynienia z wartościami
# normalnie rozłożonymi czy np. bardzo rozrzuconymi.
plt.figure(figsize=(10, 6))
sns.histplot(data['MEDV'], kde=True, bins=30)
plt.title("Rozkład ceny domów (MEDV)")
plt.xlabel("Cena domów")
plt.ylabel("Liczba domów")
plt.show()

# 2. Korelacja zmiennych z ceną domów (MEDV)
# Analizujemy korelacje między różnymi zmiennymi, aby zidentyfikować, które cechy mają najsilniejszy związek z ceną domów.
# Wartości korelacji powyżej 0.5 lub poniżej -0.5 mogą sugerować silny związek.
plt.figure(figsize=(12, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Macierz korelacji")
plt.show()

# 3. Wybór najważniejszych cech
# Wybieramy zmienne, które mają korelację powyżej 0.5 lub poniżej -0.5 w stosunku do zmiennej MEDV (cena domów).
significant_features = corr_matrix['MEDV'][corr_matrix['MEDV'].abs() > 0.5].index.tolist()
significant_features.remove('MEDV')
print(f"Najbardziej istotne cechy: {significant_features}")

# 4. Boxplot dla wybranych cech
# Wizualizujemy rozkład najważniejszych zmiennych, aby zobaczyć, czy istnieją wartości odstające (outliers).
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[significant_features])
plt.title("Boxplot dla istotnych cech")
plt.show()

# 5. Scatterplot (punktowy wykres rozproszenia)
# Tworzymy pairplot (wykresy zależności między zmiennymi), aby zobaczyć, jak różne zmienne wpływają na cenę domów.
# Może to ujawnić relacje liniowe lub nieliniowe między zmiennymi.
plt.figure(figsize=(12, 6))
sns.pairplot(data[significant_features + ['MEDV']])
plt.title("Pairplot dla zmiennych istotnych i ceny domów")
plt.show()

# 6. Standaryzacja zmiennych
# Standaryzujemy dane, aby każda zmienna miała średnią 0 i odchylenie standardowe 1.
# Jest to ważne przy regresji liniowej, ponieważ zmienne o dużych wartościach mogą dominować model.
scaler = StandardScaler()
X_std = scaler.fit_transform(data[significant_features])
X_std_df = pd.DataFrame(X_std, columns=significant_features)

# Wizualizujemy dane po standaryzacji
plt.figure(figsize=(12, 6))
sns.boxplot(data=X_std_df)
plt.title("Boxplot dla cech po standaryzacji")
plt.show()

# 7. Usuwanie odchyłów (outliers) za pomocą IQR (Interquartile Range)
# Wartości odstające mogą zaburzać model, dlatego usuwamy je za pomocą metody IQR.
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Definiowanie wartości odstających
outlier_condition = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
print("\nLiczba wartości odstających w każdej kolumnie:")
print(outlier_condition.sum())

# Usuwanie odchyłów
data_cleaned = data[~outlier_condition.any(axis=1)]
print(f"Liczba próbek po usunięciu odchyłów: {len(data_cleaned)}")

# ----- Przygotowanie danych do modelowania -----
# Tworzymy macierz X (cechy) oraz y (zmienna zależna - MEDV)
X = data_cleaned[significant_features].values
y = data_cleaned['MEDV'].values

# Standaryzacja cech
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Podział danych na zbiory treningowe (80%) i testowe (20%)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# ----- Dopasowanie modelu regresji liniowej -----
# Tworzymy model regresji liniowej i dopasowujemy go do danych treningowych.
lr = LinearRegression()
lr.fit(X_train, y_train)

# ----- Ocena modelu -----
# Predykcja wartości na zbiorze testowym
y_pred = lr.predict(X_test)

# ----- Wizualizacja wyników -----
# Wykres rzeczywiste wartości vs przewidywane wartości.
# Pokazuje, jak dobrze model dopasowuje się do danych testowych. Idealnie punkty powinny leżeć na linii y=x.
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, label="Test Data")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color="red")
plt.xlabel("Prawdziwe wartości (y_test)")
plt.ylabel("Przewidywane wartości (y_pred)")
plt.title("Regresja liniowa - Zbiór testowy")
plt.legend()
plt.show()

# ----- Dodatkowa analiza skuteczności modelu -----
# Obliczamy metryki oceny modelu: MAE (Mean Absolute Error), MSE (Mean Squared Error) i R².
# Te metryki oceniają, jak dobrze model przewiduje wartości na zbiorze testowym.
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMetryki dla zbioru testowego:\nMAE: {mae}\nMSE: {mse}\nR²: {r2}")

# Wykres residuów (różnice między rzeczywistymi a przewidywanymi wartościami).
# Pozwala zobaczyć, czy model systematycznie popełnia błędy w określonych zakresach.
residuals = y_test - y_pred

plt.figure(figsize=(7, 7))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Wykres residuów (różnice rzeczywiste - przewidywane)")
plt.xlabel("Przewidywane wartości")
plt.ylabel("Residuły")
plt.show()

# Histogram residuów - oceniamy, czy błędy modelu mają rozkład normalny.
plt.figure(figsize=(7, 7))
sns.histplot(residuals, kde=True)
plt.title("Rozkład residuów (błędy)")
plt.xlabel("Residuły")
plt.ylabel("Częstotliwość")
plt.show()

# ----- Powtarzanie analizy na danych bez odchyłów -----
# Predykcja na danych po usunięciu odchyłów
y_o_iqr_pred = lr.predict(X_test)

# Obliczamy metryki oceny dla modelu po usunięciu odchyłów.
mae_cleaned = mean_absolute_error(y_test, y_o_iqr_pred)
mse_cleaned = mean_squared_error(y_test, y_o_iqr_pred)
r2_cleaned = r2_score(y_test, y_o_iqr_pred)

print(f"\nMetryki dla danych po usunięciu odchyłów:\nMAE: {mae_cleaned}\nMSE: {mse_cleaned}\nR²: {r2_cleaned}")

# Wykres rzeczywiste wartości vs przewidywane wartości po usunięciu odchyłów.
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_o_iqr_pred, label="Test Data (po usunięciu odchyłów)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color="red")
plt.xlabel("Prawdziwe wartości")
plt.ylabel("Przewidywane wartości")
plt.title("Regresja liniowa - Zbiór po usunięciu odchyłów")
plt.legend()
plt.show()

# ----- Podsumowanie -----
# Porównanie skuteczności modelu na danych przed i po usunięciu odchyłów.
print(f"\nPorównanie wyników:\n"
      f"MAE (przed usunięciem odchyłów): {mae}\n"
      f"MAE (po usunięciu odchyłów): {mae_cleaned}\n"
      f"MSE (przed usunięciem odchyłów): {mse}\n"
      f"MSE (po usunięciu odchyłów): {mse_cleaned}\n"
      f"R² (przed usunięciem odchyłów): {r2}\n"
      f"R² (po usunięciu odchyłów): {r2_cleaned}\n")
