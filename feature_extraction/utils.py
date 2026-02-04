import os
import sqlite3
from PIL import Image
import torch
from torchvision.transforms import PILToTensor, Resize
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_max_level(sqlite_path):
    """
    Retrieves the maximal zoom level in the quad tree structure.
    """
    try:
        con = sqlite3.connect(sqlite_path)
        cursor = con.cursor()
        cursor.execute("""
                       SELECT value FROM metadata WHERE key = 'levels'
                       """)
        level = int(cursor.fetchone()[0])
        con.close()
        # Caution: levels are counted as 0,..., n-1
        return (level - 1)
    except Exception as e:
        print(f'Error reading max level from {sqlite_path}: {e}')
        return None


def get_img_sqlite(sqlite_path, level=0, size=224):
    """
    Retrieves all images and their (x, y) coordinates at a given zoom level
    from the SQLite database, resizing each tile to `size x size`.

    Returns:
      images (list of PIL.Image): List of resized tile images (length N).
      coords (list of tuple(int, int)): List of (x, y) coordinates corresponding
                                        to each tile.
      If no tiles found or an error occurs, returns ([], []) and prints a warning.
    """
    try:
        con = sqlite3.connect(sqlite_path)
    except Exception as e:
        print(f"[WARNING] Error connecting to SQLite file {sqlite_path}: {e}")
        return [], []

    try:
        cursor = con.cursor()
        cursor.execute("""
            SELECT x, y, jpeg FROM tiles WHERE level=?
        """, (level,))
        rows = cursor.fetchall()
        con.close()

        if not rows:
            # <-- key change: don't raise, just warn and return empty
            print(f"[WARNING] No images found at level {level} in {sqlite_path}. Skipping this slide.")
            return [], []

        images = []
        coords = []

        for tile_x, tile_y, jpeg_buffer in rows:
            bytes_io = BytesIO(jpeg_buffer)
            with Image.open(bytes_io) as pil_image:
                # Resize to (size, size) with a chosen resampling method
                pil_image = pil_image.resize((size, size), resample=Image.Resampling.LANCZOS)
                images.append(pil_image)
            coords.append((tile_x, tile_y))

        return images, coords

    except Exception as e:
        # <-- key change: don't raise, just warn and return empty
        print(f"[WARNING] Error reading images from {sqlite_path}: {e}")
        return [], []


def get_diagnosis(sqlite_path):
    """
    Retrieves the subtype diagnosis from the metadata table.
    """
    try:
        con = sqlite3.connect(sqlite_path)
        cursor = con.cursor()
        cursor.execute("""
            SELECT value FROM metadata WHERE key = 'diagnosis'
        """)
        row = cursor.fetchone()
        con.close()

        if row is None:
            print(f'No diagnosis found within {sqlite_path}')
            return None

        diagnosis = row[0]
        return diagnosis

    except Exception as e:
        print(f'Error reading diagnosis from {sqlite_path}: {e}')
        return None


def load_slide_data(sqlite_path, level=0, size=224):
    """
    Convenience wrapper to get tiles + coords + slide-level diagnosis.

    Returns:
      images, coords, diagnosis
    """
    images, coords = get_img_sqlite(sqlite_path, level=level, size=size)
    diagnosis = get_diagnosis(sqlite_path)
    return images, coords, diagnosis
