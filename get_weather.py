import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Chargement des données ---
df = pd.read_csv("incendies_meteo_safe.csv", delimiter=';')

# --- 2. Colonnes utilisées pour prédire la surface brûlée ---
features = [
    'temp', 'humidity', 'pressure', 'wind_speed', 'wind_deg', 'rain_1h',
    'Surface forêt (m2)', 'Surface maquis garrigues (m2)',
    'Autres surfaces naturelles hors forêt (m2)', 'Surfaces agricoles (m2)',
    'Surface autres terres boisées (m2)', 'Surfaces non boisées (m2)',
    'Nature'  # variable catégorielle
]

target = 'Surface parcourue (m2)'

for col in ['temp', 'humidity', 'pressure', 'wind_speed', 'Surface parcourue (m2)']:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

df['Nature'].fillna('Inconnue', inplace=True)

print(f"Lignes après imputation : {len(df)}")



# --- 3. Séparation X / y ---
X = df[features]
y = df[target]

# --- 4. Pipeline d'encodage + modèle ---
categorical = ['Nature']
numerical = list(set(features) - set(categorical))

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# --- 5. Split & entraînement ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# --- 6. Prédictions et évaluation ---
y_pred = model.predict(X_test)

print("✅ Régression linéaire entraînée")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# --- 7. Résidus ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.xlabel("Surface réelle (m2)")
plt.ylabel("Surface prédite (m2)")
plt.title("Surface brûlée réelle vs prédite")
plt.plot([0, y.max()], [0, y.max()], color='red', linestyle='--')
plt.tight_layout()
plt.show()
