import cv2
import numpy as np
import json
import os

def convert_image_to_grid(img_path, grid_size=(60, 30), start=(1,1), goal=(28,58), save_path="maps/dhtl_map.json"):
    """
    Chuyển ảnh bản đồ thành lưới 0/1/2/3
    0 = đường, 1 = tường, 2 = start, 3 = goal
    """
    # Đọc ảnh
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ngưỡng nhị phân: đường đi (đen) = 0, vật cản (trắng/màu) = 1
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Invert để nền đen = 0 (free), trắng = 1 (wall)
    binary = cv2.bitwise_not(binary)

    # Resize về kích thước lưới mong muốn
    grid = cv2.resize(binary, grid_size, interpolation=cv2.INTER_NEAREST)

    # Chuyển sang ma trận 0/1
    grid = (grid > 127).astype(int)

    # Đặt Start và Goal
    sr, sc = start
    gr, gc = goal
    grid[sr, sc] = 2
    grid[gr, gc] = 3

    # Lưu JSON
    data = {"grid": grid.tolist()}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Đã lưu bản đồ vào {save_path}")
    return grid

if __name__ == "__main__":
    grid = convert_image_to_grid("DHTL.png", grid_size=(60,30), start=(1,1), goal=(28,58))
    print(np.array(grid))
