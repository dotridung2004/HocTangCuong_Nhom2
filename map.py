import pygame
import numpy as np
import random
import json

ROWS, COLS = 30, 30
CELL = 20
WIDTH, HEIGHT = COLS*CELL, ROWS*CELL

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Vẽ vật cản nhiều ô với tên")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 16)

# map trống
grid = np.zeros((ROWS, COLS), dtype=int)
grid_name = np.full((ROWS, COLS), "", dtype=object)

# dict lưu màu cho từng tên vật cản
name_colors = {}
colors_bg = [
    (200,0,0), (0,200,0), (0,0,200),
    (200,200,0), (200,100,0), (150,0,150),
    (0,200,200), (255,150,150)
]
colors_text = [
    (255,255,255), (0,0,0), (255,255,255),
    (0,0,0), (255,255,255), (255,255,255),
    (0,0,0), (0,0,0)
]

used_indices = set()  # để không bị trùng màu

def get_new_color():
    for i in range(len(colors_bg)):
        if i not in used_indices:
            used_indices.add(i)
            return colors_bg[i], colors_text[i]
    # nếu hết màu, quay vòng
    return random.choice(colors_bg), random.choice(colors_text)

drawing_wall = True
selecting = False
start_pos = None
current_pos = None

# danh sách các vùng vật cản: mỗi phần tử dict {"name","r_start","r_end","c_start","c_end"}
walls = []

def draw():
    screen.fill((255,255,255))

    # vẽ các ô vật cản
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c*CELL, r*CELL, CELL, CELL)
            if grid[r,c]==1:
                name = grid_name[r,c]
                color_bg = tuple(name_colors.get(name, {}).get("bg", (0,0,0)))
            else:
                color_bg = (230,230,230)
            pygame.draw.rect(screen, color_bg, rect)
            pygame.draw.rect(screen, (180,180,180), rect, 1)

    # vẽ tên cho từng vùng
    for wall in walls:
        name = wall["name"]
        r_start, r_end = wall["r_start"], wall["r_end"]
        c_start, c_end = wall["c_start"], wall["c_end"]
        center_r = (r_start + r_end)//2
        center_c = (c_start + c_end)//2
        color_text = tuple(name_colors.get(name, {}).get("text", (255,255,255)))
        text = font.render(name, True, color_text)
        text_rect = text.get_rect(center=(center_c*CELL+CELL//2, center_r*CELL+CELL//2))
        screen.blit(text, text_rect)

    # vẽ khung vùng đang kéo
    if selecting and start_pos and current_pos:
        r1, c1 = start_pos
        r2, c2 = current_pos
        r_start, r_end = min(r1,r2), max(r1,r2)
        c_start, c_end = min(c1,c2), max(c1,c2)
        rect = pygame.Rect(c_start*CELL, r_start*CELL,
                           (c_end-c_start+1)*CELL, (r_end-r_start+1)*CELL)
        pygame.draw.rect(screen, (255,0,0), rect, 2)

    pygame.display.flip()

def save_map():
    np.save("my_map.npy", grid)
    np.save("my_map_name.npy", grid_name, allow_pickle=True)
    np.save("walls.npy", walls, allow_pickle=True)
    with open("name_colors.json", "w") as f:
        json.dump(name_colors, f)
    print("Map và thông tin vật cản đã lưu")

def load_map():
    global grid, grid_name, walls, name_colors
    grid = np.load("my_map.npy")
    grid_name = np.load("my_map_name.npy", allow_pickle=True)
    walls = np.load("walls.npy", allow_pickle=True).tolist()
    with open("name_colors.json", "r") as f:
        name_colors = json.load(f)
    print("Map đã load xong")

running = True
while running:
    draw()
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.MOUSEBUTTONDOWN:
            if e.button == 1 and drawing_wall:
                selecting = True
                start_pos = (e.pos[1]//CELL, e.pos[0]//CELL)
                current_pos = start_pos
        elif e.type == pygame.MOUSEMOTION:
            if selecting:
                current_pos = (e.pos[1]//CELL, e.pos[0]//CELL)
        elif e.type == pygame.MOUSEBUTTONUP:
            if e.button == 1 and drawing_wall and selecting:
                end_pos = (e.pos[1]//CELL, e.pos[0]//CELL)
                r1, c1 = start_pos
                r2, c2 = end_pos
                r_start, r_end = min(r1,r2), max(r1,r2)
                c_start, c_end = min(c1,c2), max(c1,c2)

                # nhập tên vật cản
                name = input(f"Nhập tên vật cản cho vùng ({r_start},{c_start}) → ({r_end},{c_end}): ")

                # nếu tên mới, gán màu không trùng
                if name not in name_colors:
                    bg, text = get_new_color()
                    name_colors[name] = {"bg": bg, "text": text}

                # gán vật cản lên grid
                for r in range(r_start, r_end+1):
                    for c in range(c_start, c_end+1):
                        grid[r,c] = 1
                        grid_name[r,c] = name

                # lưu vùng vừa tạo
                walls.append({"name": name, "r_start": r_start, "r_end": r_end,
                              "c_start": c_start, "c_end": c_end})

                selecting = False
                start_pos = None
                current_pos = None
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_w:
                drawing_wall = True
            elif e.key == pygame.K_e:
                drawing_wall = False
            elif e.key == pygame.K_l:
                save_map()
            elif e.key == pygame.K_o:  # load map bằng phím O
                load_map()

    clock.tick(30)

pygame.quit()