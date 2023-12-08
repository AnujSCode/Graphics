import numpy as np
import pygame
import time
from concurrent.futures import ThreadPoolExecutor

class Object3D:
    def __init__(self, center):
        self.center = np.array(center)

class Sphere(Object3D):
    def __init__(self, center, radius):
        super().__init__(center)
        self.radius = radius

    def sdf(self, pos):
        return np.linalg.norm(pos - self.center) - self.radius
class Triangle(Object3D):
    def __init__(self, v0, v1, v2):
        super().__init__(v0)
        self.v0 = np.array(v0)
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.normal = np.cross(self.v1 - self.v0, self.v2 - self.v0)

    def sdf(self, pos):
        edge0 = self.v1 - self.v0
        edge1 = self.v2 - self.v1
        edge2 = self.v0 - self.v2
        v0 = pos - self.v0
        v1 = pos - self.v1
        v2 = pos - self.v2
        sq_edge0 = np.dot(edge0, edge0)
        sq_edge1 = np.dot(edge1, edge1)
        sq_edge2 = np.dot(edge2, edge2)
        dot_v0_edge0 = np.dot(v0, edge0)
        dot_v1_edge1 = np.dot(v1, edge1)
        dot_v2_edge2 = np.dot(v2, edge2)
        region0 = dot_v2_edge2 * sq_edge0 - dot_v0_edge0 * np.dot(edge0, edge2)
        region1 = dot_v0_edge0 * sq_edge1 - dot_v1_edge1 * np.dot(edge0, edge1)
        region2 = dot_v1_edge1 * sq_edge2 - dot_v2_edge2 * np.dot(edge1, edge2)
        if region0 < 0 and region1 < 0 and region2 < 0:
            return np.dot(v0, self.normal) * np.dot(v0, self.normal) / np.dot(self.normal, self.normal)
        else:
            epsilon = 1e-9
            dist0 = max(dot_v0_edge0 * dot_v0_edge0 / sq_edge0 - np.dot(v0, v0), 0) + epsilon
            dist1 = max(dot_v1_edge1 * dot_v1_edge1 / sq_edge1 - np.dot(v1, v1), 0) + epsilon
            dist2 = max(dot_v2_edge2 * dot_v2_edge2 / sq_edge2 - np.dot(v2, v2), 0) + epsilon
            return np.sqrt(min(dist0, dist1, dist2))



class Cube(Object3D):
    def __init__(self, center, side):
        super().__init__(center)
        self.side = side

    def sdf(self, pos):
        q = abs(pos - self.center) - self.side / 2
        return np.linalg.norm(np.maximum(q, 0)) + min(max(q[0], max(q[1], q[2])), 0)
class Renderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def raymarch(self, scene, ray_origin, ray_direction):
        depth = 0.0
        max_depth = 100.0
        min_distance = 0.001
        color = np.zeros(3)

        while depth < max_depth:
            pos = ray_origin + depth * ray_direction
            distance = min([shape.sdf(pos) for shape in scene])

            if distance < min_distance:
                color = [255, 255, 255]
                break

            depth += distance

        return color

    def render_pixel(self, x, y, scene):
        ndc_x = (2 * x - self.width) / self.height
        ndc_y = (2 * y - self.height) / self.height

        ray_origin = np.array([0, 0, 0])
        ray_direction = np.array([ndc_x, ndc_y, 1.0])

        color = self.raymarch(scene, ray_origin, ray_direction)
        return self.height - y - 1, x, color

    def render(self):
        scene = [Sphere(center=[1, 0, 10], radius=0.75),
                 Cube(center=[-1, 0, 5], side=.5)

                 ]
        image = np.zeros((self.height, self.width, 3))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.render_pixel, x, y, scene) for y in range(self.height) for x in
                       range(self.width)]
            for future in futures:
                y, x, color = future.result()
                image[y, x] = color

        return image

def main():
    w, h = 512, 512
    renderer = Renderer(w, h)

    pygame.init()
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

    start_time = time.time()
    print("Rendering started...")
    rendered_image = renderer.render()
    end_time = time.time()
    print(f"Rendering finished. It took {end_time - start_time} seconds.")

    pygame_surface = pygame.surfarray.make_surface(rendered_image)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.blit(pygame_surface, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()