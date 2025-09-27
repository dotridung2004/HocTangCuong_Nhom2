import pygame
import numpy as np
import json
from env.gridworld import GridWorld
from algorithms.policy_iteration import policy_iteration

# ----- Load map đã lưu -----
grid = np.load("my_map.npy")
grid_name = np.load("my_map_name.npy", allow_pickle=True)
walls = np.load("walls.npy", allow_pickle=True).tolist()
with open("name_colors.json", "r") as f:
    name_colors = json.load(f)

ROWS, COLS = grid.shape
CELL = 20
WIDTH, HEIGHT = COLS*CELL, ROWS*CELL

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Policy Iteration từ map đã lưu")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 16)  # font để vẽ tên vật cản

start = None
goal = None
path = []

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

    # vẽ start và goal
    if start:
        pygame.draw.rect(screen,(0,255,0),
                         pygame.Rect(start[1]*CELL,start[0]*CELL,CELL,CELL))
    if goal:
        pygame.draw.rect(screen,(255,165,0),
                         pygame.Rect(goal[1]*CELL,goal[0]*CELL,CELL,CELL))

    # vẽ path
    for p in path:
        pygame.draw.rect(screen,(0,0,255),
                         pygame.Rect(p[1]*CELL,p[0]*CELL,CELL,CELL))

    pygame.display.flip()

def reconstruct(policy_map):
    pth = []
    state = start
    while state != goal:
        a = policy_map[state]
        if a==0: nxt = (max(state[0]-1,0), state[1])
        elif a==1: nxt = (min(state[0]+1,ROWS-1), state[1])
        elif a==2: nxt = (state[0], max(state[1]-1,0))
        else:      nxt = (state[0], min(state[1]+1,COLS-1))
        if nxt==state: break
        pth.append(nxt)
        state = nxt
    return pth

running = True
while running:
    draw()
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.MOUSEBUTTONDOWN:
            r, c = e.pos[1]//CELL, e.pos[0]//CELL
            if e.button == 1 and grid[r,c]==0:      # click trái chọn start/goal
                if start is None:
                    start = (r,c)
                elif goal is None:
                    goal = (r,c)
            elif e.button == 3 and grid[r,c]==0:    # click phải xóa start/goal
                if (r,c)==start: start=None
                elif (r,c)==goal: goal=None
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE and start and goal:
                env = GridWorld(grid, start, goal)
                V, pol = policy_iteration(env)
                policy_map = {(i,j): pol[i,j] for i in range(ROWS) for j in range(COLS)}
                path = reconstruct(policy_map)
            if e.key == pygame.K_c:  # clear start, goal, path
                start=None; goal=None; path=[]
    clock.tick(30)

pygame.quit()