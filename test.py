import numpy as np
import matplotlib.pyplot as plt

# Define a class to represent a Sphere
class Sphere:
    def __init__(self, center, radius, color, material):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.material = material

    def intersect(self, ray_origin, ray_direction):
        oc = ray_origin - self.center
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return -1  # No intersection

        # Return the nearest intersection point
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        return t if t > 0 else -1

# Phong shading
def phong_shading(hit_point, normal, view_direction, material, lights, spheres):
    ambient = 0.1  # Ambient light intensity
    diffuse = specular = 0

    for light in lights:
        if light['type'] == 'point':
            light_direction = np.array(light['position']) - hit_point
            light_distance = np.linalg.norm(light_direction)
            light_direction = light_direction / light_distance
            attenuation = 1.0 / (light_distance ** 2)

        elif light['type'] == 'directional':
            light_direction = np.array(light['direction'])
            attenuation = 1.0  # Directional light has constant intensity

        # Diffuse component
        diffuse += (
            material['diffuse']
            * max(np.dot(normal, light_direction), 0)
            * attenuation
        )

        # Specular component
        reflection = 2 * np.dot(normal, light_direction) * normal - light_direction
        specular += (
            material['specular']
            * max(np.dot(view_direction, reflection), 0) ** 16  # Shininess factor (here 16)
            * attenuation
        )

    return ambient + diffuse + specular

# Recursive ray tracing
def recursive_ray_trace(ray_origin, ray_direction, spheres, depth):
    if depth <= 0:
        return [0, 0, 0]  # Terminate recursion at a certain depth

    closest_sphere = None
    closest_distance = float('inf')

    for sphere in spheres:
        t = sphere.intersect(ray_origin, ray_direction)
        if t > 0 and t < closest_distance:
            closest_distance = t
            closest_sphere = sphere

    if closest_sphere:
        hit_point = ray_origin + ray_direction * closest_distance
        normal = (hit_point - closest_sphere.center) / closest_sphere.radius
        view_direction = -ray_direction

        # Phong shading calculation
        phong_color = phong_shading(hit_point, normal, view_direction, closest_sphere.material, lights, spheres)

        # Recursive reflection for shiny objects
        if closest_sphere.material['specular'] > 0:
            reflection_direction = ray_direction - 2 * np.dot(ray_direction, normal) * normal
            reflection_color = recursive_ray_trace(hit_point, reflection_direction, spheres, depth - 1)
            return phong_color + np.array(reflection_color) * closest_sphere.material['specular']
        else:
            return phong_color

    else:
        return [0, 0, 0]  # Background color


# Parse the scene file
def parse_scene(file_path):
    spheres = []
    lights = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.split()
            if tokens[0] == 'image':
                viewport_width = int(tokens[1])
                viewport_height = int(tokens[2])
            elif tokens[0] == 'viewport':
                viewplane = list(map(float, tokens[1:]))
            elif tokens[0] == 'eye':
                eye = np.array(list(map(float, tokens[1:])))
            elif tokens[0] == 'sphere':
                center = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
                radius = float(tokens[4])
                color = [int(tokens[5]), int(tokens[6]), int(tokens[7])]
                material = {'diffuse': float(tokens[8]), 'specular': float(tokens[9])}
                spheres.append(Sphere(center, radius, color, material))
            elif tokens[0] == 'light':
                light_type = tokens[1]
                if light_type == 'point':
                    position = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
                    color = [float(tokens[5]), float(tokens[6]), float(tokens[7])]
                    lights.append({'type': 'point', 'position': position, 'color': color})
                elif light_type == 'directional':
                    direction = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
                    color = [float(tokens[5]), float(tokens[6]), float(tokens[7])]
                    lights.append({'type': 'directional', 'direction': direction, 'color': color})

    return viewport_width, viewport_height, viewplane, eye, spheres, lights

# Ray tracing function
def ray_trace(ray_origin, ray_direction, spheres):
    closest_sphere = None
    closest_distance = float('inf')

    for sphere in spheres:
        t = sphere.intersect(ray_origin, ray_direction)
        if t > 0 and t < closest_distance:
            closest_distance = t
            closest_sphere = sphere

    if closest_sphere:
        return closest_sphere.color
    else:
        return [0, 0, 0]  # Background color

# Scene setup and rendering
scene_file = 'scene3.txt'  # Replace with your scene file path
spheres, lights = parse_scene(scene_file)

# Scene setup and rendering
# ... (Parsing scene file and setting up the scene)

# Array to store pixel colors
image = np.zeros((viewport_height, viewport_width, 3), dtype=np.uint8)

# Ray tracing loop for each pixel in the viewport
for y in range(viewport_height):
    for x in range(viewport_width):
        ray_direction = np.array([
            viewplane[0] + (viewplane[2] - viewplane[0]) * (x + 0.5) / viewport_width,
            viewplane[1] + (viewplane[3] - viewplane[1]) * (y + 0.5) / viewport_height,
            viewplane[2]
        ]) - eye
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        pixel_color = recursive_ray_trace(eye, ray_direction, spheres, depth=3)
        image[y, x] = np.clip(pixel_color, 0, 255)  # Store color in the image array

# Display the rendered image using Matplotlib
plt.imshow(image)
plt.axis('off')  # Hide axis ticks
plt.show()