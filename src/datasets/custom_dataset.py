import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomImageDatasetFromCSV(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)  # Membaca file CSV
        self.img_dir = img_dir  # Direktori gambar
        self.transform = transform  # Transformasi yang akan diterapkan pada gambar
        self.classes = sorted(self.data_frame[['jenis', 'warna']].drop_duplicates().apply(tuple, axis=1))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data_frame)  # Mengembalikan jumlah data

    def __getitem__(self, idx):
        img_id = self.data_frame.iloc[idx, 0]  # Mengambil ID gambar dari kolom pertama
        label = self.data_frame.iloc[idx, 1]  # Mengambil label dari kolom kedua

        # Cek keberadaan gambar dengan dua format: .jpg dan .png
        img_path_jpg = os.path.join(self.img_dir, f"{img_id}.jpg")
        img_path_png = os.path.join(self.img_dir, f"{img_id}.png")

        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            raise FileNotFoundError(f"Image not found: {img_id} (both .jpg and .png)")

        # Membaca gambar
        image = Image.open(img_path).convert("RGB")  # Mengubah ke format RGB

        # Menerapkan transformasi jika ada
        if self.transform:
            image = self.transform(image)

        return image, label  # Mengembalikan gambar dan label
