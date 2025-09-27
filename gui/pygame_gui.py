import pygame

def run_gui(env, path):
    pygame.init()
    cell_size = 80
    width = env.grid.shape[1] * cell_size
    height = env.grid.shape[0] * cell_size

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Robot Pathfinding")

    colors = {
        0: (255, 255, 255),
        1: (0, 0, 0),
        "start": (0, 255, 0),
        "goal": (255, 0, 0),
        "path": (0, 0, 255)
    }

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for i in range(env.grid.shape[0]):
            for j in range(env.grid.shape[1]):
                rect = pygame.Rect(j*cell_size, i*cell_size, cell_size, cell_size)
                color = colors[env.grid[i, j]]
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (200,200,200), rect, 1)

        si, sj = env.start
        gi, gj = env.goal
        pygame.draw.rect(screen, colors["start"], pygame.Rect(sj*cell_size, si*cell_size, cell_size, cell_size))
        pygame.draw.rect(screen, colors["goal"], pygame.Rect(gj*cell_size, gi*cell_size, cell_size, cell_size))

        for (x,y) in path:
            rect = pygame.Rect(y*cell_size, x*cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, colors["path"], rect)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
