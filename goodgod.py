#!/usr/bin/env python3

import numpy as np
from PIL import Image
import multiprocessing
from multiprocessing import Pool
import pygame

scene_file = "scene3.txt"


class Light:
    def __init__(self, position, color, type):
        self.position = position
        self.color = color
        self.type = type  # 'point' or 'directional'
class Object:
    def __init__(self, color, position = None, normal = None):
        self.color = color if color is not None else [225, 225, 225]
        self.position = position if position is not None else [0, 0, 0]  # default position
        self.normal = normal if normal is not None else [0, 1, 0]  # default normal is


class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)

class Intersection:
    def __init__(self, t, point, object):
        self.t = t
        self.point = point
        self.object = object

class Sphere(Object):
    def __init__(self, center, radius, color):
        super().__init__(color)
        self.center = center
        self.radius = radius

    def intersect(self, ray):
        # Compute the quadratic parameters
        oc = np.subtract(ray.origin, self.center)
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius

        # Solve the quadratic equation
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None  # No intersection
        else:
            t = (-b - np.sqrt(discriminant)) / (2.0 * a)
            point = np.add(ray.origin, np.multiply(t, ray.direction))
            return Intersection(t, point, self)  # Return the intersection point and the object

    def transform(self, transformation):
        # Transform the sphere by applying the transformation matrix to its center
        center_homogeneous = np.append(self.center, 1)  # Convert to homogeneous coordinates
        transformed_center_homogeneous = np.dot(transformation.get_matrix(), center_homogeneous)
        self.center = transformed_center_homogeneous[:3]  # Convert back to 3D coordinates

    def get_normal(self, point):
        return normalize(point - self.center)


class Plane(Object):
    def __init__(self, position, normal, color):
        super().__init__(color)
        self.position = position if position is not None else [0, -10, 0]  # default position is origin
        self.normal = normal if normal is not None else [0, 1, 0]  # default normal is upward

    def intersect(self, ray):
        denom = np.dot(ray.direction, self.normal)
        if np.abs(denom) > 1e-6:
            t = np.dot(np.subtract(self.position, ray.origin), self.normal) / denom
            if t >= 0:
                point = np.add(ray.origin, np.multiply(t, ray.direction))
                return Intersection(t, point, self)  # Return the intersection point and the object
        return None

    def get_normal(self, _):
        return self.normal

class Triangle(Object):
    def __init__(self, p1, p2, p3, color):
        super().__init__(color)
        self.p1 = np.array([p1[0], p1[1], 0])  # Assuming vertices are in 2D, adding a default z-value of 0
        self.p2 = np.array([p2[0], p2[1], 0])
        self.p3 = np.array([p3[0], p3[1], 0])

    def intersect(self, ray):
        # Compute vectors along two edges of the triangle
        edge1 = self.p2 - self.p1
        edge2 = self.p3 - self.p1

        # Begin calculating determinant - also used to calculate U parameter
        pvec = np.cross(ray.direction, edge2)

        # If determinant is near zero, ray lies in plane of triangle
        det = np.dot(edge1, pvec)

        # NOT CULLING
        if det > -0.000001 and det < 0.000001:
            return None
        inv_det = 1.0 / det

        # Calculate distance from V1 to ray origin
        tvec = ray.origin - self.p1

        # Calculate U parameter and test bound
        u = np.dot(tvec, pvec) * inv_det
        if u < 0.0 or u > 1.0:
            return None

        # Prepare to test V parameter
        qvec = np.cross(tvec, edge1)

        # Calculate V parameter and test bound
        v = np.dot(ray.direction, qvec) * inv_det
        if v < 0.0 or u + v > 1.0:
            return None

        t = np.dot(edge2, qvec) * inv_det

        if t > 0.000001:  # ray intersection
            return Intersection(t, ray.origin + ray.direction * t, self)

        # No hit, no win
        return None

    def get_normal(self, _):
        # The normal is constant for a flat triangle, so we don't need the intersection point
        edge1 = self.p2 - self.p1
        edge2 = self.p3 - self.p1
        return normalize(np.cross(edge1, edge2))

class Scene:
    def __init__(self, viewport, image_size, camera):
        self.viewport = viewport
        self.image_size = image_size
        self.camera = camera
        self.objects = []
        self.lights = []

    def add_object(self, obj):
        self.objects.append(obj)

    def add_light(self, light):
        self.lights.append(light)


def read_scene_file(scene_file):
    scene = None
    with open(scene_file, 'r') as f:
        for line in f:
            tokens = line.split()
            if not tokens:  # skip empty lines
                continue
            if tokens[0] == 'image':
                # create a new Scene object
                scene = Scene(None, (int(tokens[1]), int(tokens[2])), None)
            elif tokens[0] == 'viewport':
                # set the viewport of the Scene
                scene.viewport = list(map(float, tokens[1:]))
            elif tokens[0] == 'eye':
                # set the camera of the Scene
                scene.camera = list(map(float, tokens[1:4]))  # ensure it's a 3-element list
            elif tokens[0] == 'sphere':
                # add a Sphere to the Scene
                scene.add_object(Sphere(list(map(float, tokens[1:4])), float(tokens[4]), (255, 0, 0)))  # ensure center is a 3-element list
            elif tokens[0] == 'plane':
                # add a Plane to the Scene
                scene.add_object(Plane(list(map(float, tokens[1:4])), None, (0, 255, 0)))  # ensure position is a 3-element list
            elif tokens[0] == 'triangle':
                # add a Triangle to the Scene
                p1 = list(map(float, tokens[1:4]))
                p2 = list(map(float, tokens[4:7]))
                p3 = list(map(float, tokens[7:10]))
                color = (0, 0, 255)  # replace with actual color if available
                scene.add_object(Triangle(p1, p2, p3, color))
            elif tokens[0] == 'light':
                # add a Light to the Scene
                type = tokens[1]
                position = list(map(float, tokens[2:5]))
                color = (255, 255, 255)  # replace with actual color if available
                scene.add_light(Light(position, color, type))
    return scene

def render_pixel(i, j, scene):
    x = scene.viewport[0] + (scene.viewport[2] - scene.viewport[0]) * j / scene.image_size[0]
    y = scene.viewport[1] + (scene.viewport[3] - scene.viewport[1]) * i / scene.image_size[1]
    ray = Ray(scene.camera, [x, y, scene.viewport[4]])

    closest_intersection = None
    for obj in scene.objects:
        intersection = obj.intersect(ray)
        if intersection is not None and (closest_intersection is None or intersection.t < closest_intersection.t):
            closest_intersection = intersection

    if closest_intersection is not None:
        color = np.zeros(3)
        for light in scene.lights:
            # Check for shadows
            shadow_ray = Ray(closest_intersection.point, light.position - closest_intersection.point)
            shadow_intersection = None
            for obj in scene.objects:
                intersection = obj.intersect(shadow_ray)
                if intersection is not None and (shadow_intersection is None or intersection.t < shadow_intersection.t):
                    shadow_intersection = intersection
            if shadow_intersection is not None:
                continue  # Skip this light source, it's blocked by another object

            # Calculate lighting using the Phong model
            color += phong_model(closest_intersection, light, scene.camera, 0.1, 1.0, 10.0)

        return (i, j, np.clip(color, 0, 255))
    return (i, j, [0, 0, 0])  # Background color if no intersection





def render(scene):
    image = np.zeros((scene.image_size[1], scene.image_size[0], 3), dtype=np.uint8)

    # Create a thread pool with the number of workers equal to the number of CPU cores
    num_cores = multiprocessing.cpu_count()
    with Pool(2*num_cores) as p:
        results = p.starmap(render_pixel, [(i, j, scene) for i in range(scene.image_size[1]) for j in range(scene.image_size[0])])

    # Set the pixel colors in the image based on the results
    for result in results:
        i, j, color = result
        image[i, j] = color

    # Save the image to a file
    Image.fromarray(image).save('output.png')

    # Display the image using Pygame
    pygame.init()
    screen = pygame.display.set_mode(scene.image_size)
    pygame.surfarray.blit_array(screen, image)
    pygame.display.flip()

    # Wait for the user to close the window
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()


def phong_model(intersection, light, viewer, ambient_intensity, light_intensity, shininess):
    # Calculate vectors
    normal = intersection.object.get_normal(intersection.point)
    light_dir = normalize(light.position - intersection.point)
    viewer_dir = normalize(viewer - intersection.point)

    # Ambient component
    ambient = np.array(intersection.object.color) * ambient_intensity


    # Diffuse component
    diffuse_light_intensity = np.maximum(np.dot(light_dir, normal), 0) * light_intensity
    diffuse = np.array(intersection.object.color) * diffuse_light_intensity

    # Specular component
    reflection_dir = reflect(-light_dir, normal)
    specular_light_intensity = light_intensity * (np.maximum(np.dot(reflection_dir, viewer_dir), 0) ** shininess)
    specular = np.array(light.color) * specular_light_intensity

    # Sum up to get final color
    color = ambient + diffuse + specular

    return color

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflect(vector, axis):
    vector = np.array(vector)
    axis = np.array(axis)
    return vector - 2 * np.dot(vector, axis) * axis

if __name__ == "__main__":
    # Read the scene file
    scene = read_scene_file("scene3.txt")

    # Render the scene
    render(scene)
