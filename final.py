import numpy as np
import pygame

class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = color

    def distance(self, p):
        return np.linalg.norm(p - self.center) - self.radius

class Triangle:
    def __init__(self, vertices, color):
        self.vertices = [np.array(vertex) for vertex in vertices]
        self.color = color

    def distance(self, p):
        edge1 = self.vertices[1] - self.vertices[0]
        edge2 = self.vertices[2] - self.vertices[0]
        normal = np.cross(edge1, edge2)
        normal /= np.linalg.norm(normal)

        d = np.dot(normal, self.vertices[0])
        t = (np.dot(normal, p) - d) / np.dot(normal, normal)
        projection = p - t * normal

        if point_in_triangle(projection, self.vertices):
            return np.linalg.norm(projection - p)
        else:
            return np.inf

def point_in_triangle(p, triangle):
    v0 = triangle[1] - triangle[0]
    v1 = triangle[2] - triangle[0]
    v2 = p - triangle[0]

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    return (u >= 0) and (v >= 0) and (u + v < 1)

class Box:
    def __init__(self, min_corner, max_corner, color):
        self.min_corner = np.array(min_corner)
        self.max_corner = np.array(max_corner)
        self.color = color

    def distance(self, p):
        dx = max(self.min_corner[0] - p[0], 0, p[0] - self.max_corner[0])
        dy = max(self.min_corner[1] - p[1], 0, p[1] - self.max_corner[1])
        dz = max(self.min_corner[2] - p[2], 0, p[2] - self.max_corner[2])

        return np.sqrt(dx * dx + dy * dy + dz * dz)

class Scene:
    def __init__(self):
        self.objects = []

    def add_object(self, obj):
        self.objects.append(obj)

    def distance_to_scene(self, point):
        return min(obj.distance(point) for obj in self.objects)

def read_scene_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    scene = Scene()
    lights = []

    for line in lines:
        tokens = line.split()

        if not tokens:
            continue

        if tokens[0] == 'sphere':
            center = list(map(float, tokens[1:4]))
            radius = float(tokens[4])
            color = list(map(int, tokens[5:8]))
            scene.add_object(Sphere(center, radius, color))

        elif tokens[0] == 'triangle':
            vertices = [list(map(float, tokens[i:i+3])) for i in range(1, 10, 3)]
            color = list(map(int, tokens[10:13]))
            scene.add_object(Triangle(vertices, color))

        elif tokens[0] == 'box':
            min_corner = list(map(float, tokens[1:4]))
            max_corner = list(map(float, tokens[4:7]))
            color = list(map(int, tokens[7:10]))
            scene.add_object(Box(min_corner, max_corner, color))

        elif tokens[0] == 'light':
            if tokens[1] == 'directional':
                direction = list(map(float, tokens[2:5]))
                color = list(map(int, tokens[5:8]))
                lights.append(('directional', direction, color))
            elif tokens[1] == 'point':
                position = list(map(float, tokens[2:5]))
                color = list(map(int, tokens[5:8]))
                lights.append(('point', position, color))

    return scene, lights

def ray_march_pixel(i, j, viewport, image_size, camera, scene, lights):
    x = viewport[0] + (viewport[2] - viewport[0]) * j / image_size[0]
    y = viewport[1] + (viewport[3] - viewport[1]) * i / image_size[1]
    direction = [x, y, viewport[4]]

    depth = 0
    max_depth = 100  # Set a maximum depth to prevent infinite looping
    min_distance = 0.001  # Adjust this threshold as needed

    for _ in range(max_depth):
        point = np.array(camera) + depth * np.array(direction)
        dist = scene.distance_to_scene(point)
        if dist < min_distance:
            return 1 - depth / max_depth
        depth += dist

    return 0  # Return black for no intersection

def render(width, height, fov, scene, lights):
    image = np.zeros((height, width))
    viewport = [-10, -10, 10, 10, 6.394]  # Update with your viewport values
    camera = [0, 0, 5]  # Update with your camera position

    for i in range(height):
        for j in range(width):
            pixel_value = ray_march_pixel(i, j, viewport, (width, height), camera, scene, lights)
            image[i, j] = pixel_value

    pygame_image = pygame.surfarray.make_surface((255 * image).astype(np.uint8))
    return pygame_image

if __name__ == "__main__":
    # Define the camera parameters
    width, height = 512, 512
    fov = np.pi / 2

    # Read the scene file
    scene, lights = read_scene_file('finalScene.txt')

    # Render the image
    pygame.init()
    pygame_image = render(width, height, fov, scene, lights)

    # Display the image in a Pygame window
    screen = pygame.display.set_mode((width, height))
    screen.blit(pygame_image, (0, 0))
    pygame.display.flip()

    # Wait for the user to close the window
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
