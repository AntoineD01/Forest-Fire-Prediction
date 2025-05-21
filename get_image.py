import os
import requests
import pandas as pd

# Paths
base_folder = r"C:\Users\Antoine Dupont\Documents\!EFREI\OneDrive - Efrei\!Cours\Machine Learning\Project\Forest-Fire-Prediction"
images_folder = os.path.join(base_folder, "fire_images")
os.makedirs(images_folder, exist_ok=True)

csv_path = os.path.join(base_folder, "incendies_meteo_safe.csv")
output_csv_path = os.path.join(base_folder, "df_nature_with_images.csv")

# Load data
df = pd.read_csv(csv_path, delimiter=';')

# Drop irrelevant columns
cols_to_drop = [
    'weather', 'clouds', 'Précision de la donnée', 
    'Nombre de bâtiments partiellement détruits',
    'Nombre de décès', 'Nombre de bâtiments totalement détruits',
    'Autres surfaces (m2)', 'Surfaces agricoles (m2)',
    'Décès ou bâtiments touchés'
]
df_clean = df.drop(columns=cols_to_drop)
df_clean = df_clean[df_clean['Surface parcourue (m2)'] > 0].copy()
df_nature = df_clean.dropna(subset=['Nature']).copy()

# Fill missing numeric values
num_cols = df_nature.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Surface parcourue (m2)' in num_cols:
    num_cols.remove('Surface parcourue (m2)')  # exclude target column
for col in num_cols:
    median_val = df_nature[col].median()
    df_nature[col] = df_nature[col].fillna(median_val)

# Encode categorical variable
df_nature['Nature_encoded'] = df_nature['Nature'].astype('category').cat.codes

# Create date string
df_nature['date_str'] = pd.to_datetime(df_nature['Date de première alerte']).dt.strftime('%Y-%m-%d')

# Function to download image
def save_fire_image(lat, lon, date, idx, folder=images_folder, width=800, height=800, margin=0.05):
    bbox = f"{lat - margin},{lon - margin},{lat + margin},{lon + margin}"
    url = (
        f"https://wvs.earthdata.nasa.gov/api/v1/snapshot?"
        f"REQUEST=GetSnapshot&BBOX={bbox}"
        f"&CRS=EPSG:4326&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor"
        f"&WRAP=day&FORMAT=image/png"
        f"&WIDTH={width}&HEIGHT={height}&TIME={date}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        filename = f"fire_{idx}_{lat:.4f}_{lon:.4f}_{date}.png"
        filepath = os.path.join(folder, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Saved image: {filename}")
        return filepath
    else:
        print(f"Error {response.status_code} for idx={idx} at {lat},{lon} on {date}")
        return None

# Load previous CSV if exists to continue
if os.path.exists(output_csv_path):
    df_nature = pd.read_csv(output_csv_path)
    if 'image_path' not in df_nature.columns:
        df_nature['image_path'] = None
else:
    df_nature['image_path'] = None

# Loop through rows and download images
for idx, row in df_nature.iterrows():
    if pd.notnull(row['image_path']) and row['image_path'] != '':
        print(f"Skipping idx {idx}, image already saved.")
        continue

    lat = row['latitude']
    lon = row['longitude']
    date = row['date_str']

    image_path = save_fire_image(lat, lon, date, idx)
    df_nature.at[idx, 'image_path'] = image_path if image_path else ''

    # Try saving CSV
    try:
        # Convert all image_path to strings and replace 'None' with empty string
        df_nature['image_path'] = df_nature['image_path'].astype(str).replace({'None': ''})
        df_nature.to_csv(output_csv_path, index=False)
    except Exception as e:
        print(f"Error saving CSV at idx {idx}: {e}")
        break  # Optional: stop loop to fix problem 