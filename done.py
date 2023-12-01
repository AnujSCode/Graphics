import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define functions for Phong illumination model calculations
#!/usr/bin/env python3

import numpy as np
from PIL import Image
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pygame


scene_file = "scene3.txt"
class Scene:
    def __init__(self, viewport, image_size, camera):
        self.viewport = viewport
        self.image_size = image_size
        self.camera = camera
        self.objects = []
        self.lights = []

    def add_object(self, obj, material=None):
        if material is not None:
            obj.material = material
        self.objects.append(obj)


    # Add a method to add lights to the scene
    def add_light(self, light):
        self.lights.append(light)

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)
class Intersection:
    def __init__(self, t, point, object):
        self.t = t
        self.point = point
        self.object = object
class Light:
    def __init__(self, light_type, position, direction, color):
        self.type = light_type
        self.position = position if light_type == 'point' else None
        self.direction = direction if light_type == 'directional' else None
        self.color = color


# Update the Object class to include material properties
class Object:
    def __init__(self, color, position=None, normal=None, material=None, gradient_colors=None):
        self.color = color if color is not None else [225, 225, 225]
        self.position = position if position is not None else [0, 0, 0]
        self.normal = normal if normal is not None else [0, 1, 0]
        self.material = material if material is not None else {}
        self.gradient_colors = gradient_colors if gradient_colors is not None else [self.color]  # Define gradient colors

class Sphere(Object):
    def __init__(self, center, radius, color):
        super().__init__(color=color)
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

    def intersect_shadow_ray(self, ray, light_position):
        oc = self.center - ray.origin
        t_ca = np.dot(oc, ray.direction)
        d_squared = np.dot(oc, oc) - t_ca * t_ca

        # Check if the shadow ray misses the sphere
        if d_squared > self.radius * self.radius:
            return False

        t_hc = np.sqrt(self.radius * self.radius - d_squared)
        t = t_ca - t_hc if t_ca - t_hc > 0 else t_ca + t_hc

        # Check if the intersection is between the light and the shaded point
        if t > 0:
            point = ray.origin + t * ray.direction
            distance_to_light = np.linalg.norm(light_position - point)
            if distance_to_light < np.linalg.norm(light_position - ray.origin):
                return True  # Intersection is between light and shaded point

        return False  # No intersection or not between light and shaded point




class Plane(Object):
    def __init__(self, position, normal, color):
        super().__init__(color=color)
        self.position = position if position is not None else [0, 0, 0]  # default position is origin
        self.normal = normal if normal is not None else [0, 1, 0]  # default normal is upward

    def intersect(self, ray):
        denom = np.dot(ray.direction, self.normal)
        if abs(denom) > 1e-6:  # Ensure the ray is not parallel to the plane
            t = np.dot(np.subtract(self.position, ray.origin), self.normal) / denom
            if t > 0:  # Ensure the intersection is in front of the ray's origin
                point = np.add(ray.origin, np.multiply(t, ray.direction))
                return Intersection(t, point, self)  # Return the intersection point and the object
        return None

    def intersect_shadow_ray(self, ray, light_position):
        denom = np.dot(ray.direction, self.normal)
        if abs(denom) > 1e-6:  # Ensure the ray is not parallel to the plane
            t = np.dot(np.subtract(self.position, ray.origin), self.normal) / denom
            if t > 0:  # Ensure the intersection is in front of the ray's origin
                point = np.add(ray.origin, np.multiply(t, ray.direction))
                # Check if the intersection point is between the light and the shaded point
                light_direction = np.subtract(light_position, point)
                distance_to_light = np.linalg.norm(light_direction)
                # Check if the intersection point is closer to the light source than the shaded point
                if distance_to_light < np.linalg.norm(np.subtract(light_position, ray.origin)):
                    return True  # Intersection is between light and shaded point
        return False  # No intersection or not between light and shaded point

class Triangle(Object):
    def __init__(self, p1, p2, p3, color):
        super().__init__(color=color)
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)

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
def phong_lighting(intersection, lights_point, lights_directional, view_direction, material, objects):
    ambient_color = np.array([30, 30, 30], dtype=np.float64)  # Ambient color

    diffuse_color = np.array(material.get('diffuse_color', [255, 255, 255]), dtype=np.float64)
    specular_color = np.array(material.get('specular_color', [255, 255, 255]), dtype=np.float64)
    shininess = 10  # Reduced shininess for a smaller specular highlight

    object_color = np.array(intersection.object.color)  # Object color

    # Calculate ambient component
    ambient_component = ambient_color * object_color

    # Initialize final color with ambient component
    final_color = ambient_component

    # Calculate diffuse and specular components for each point light
    for light in lights_point:
        light_direction = light.position - intersection.point
        light_distance = np.linalg.norm(light_direction)
        light_direction = light_direction / light_distance  # Normalize light direction

        # Check if the point is in shadow for point lights
        in_shadow_point = is_point_in_shadow(intersection.point, lights_point, lights_directional, objects, intersection.point)


        if in_shadow:
            continue

        # Diffuse component calculation for point lights
        cos_theta = max(np.dot(intersection.object.normal, light_direction), 0)
        diffuse_component = diffuse_color * light.color * cos_theta

        # Specular component calculation for point lights
        light_direction_np = np.array(light_direction)
        intersection_normal_np = np.array(intersection.object.normal)

        reflection_direction = 2 * np.dot(light_direction_np,
                                          intersection_normal_np) * intersection_normal_np - light_direction_np

        cos_alpha = max(np.dot(reflection_direction, view_direction), 0)
        reduced_specular_color = specular_color * 0.2  # Adjust the multiplier as needed
        specular_component = reduced_specular_color * light.color * pow(cos_alpha, shininess)

        # Add diffuse and specular components for point lights to final color
        final_color += diffuse_component + specular_component

    # Calculate diffuse and specular components for each directional light
    for light in lights_directional:
        # Directional lights do not cast shadows, so proceed with calculations
        light_direction = -light.direction  # Directional light's direction
        light_distance = float('inf')  # Define an arbitrary distance for directional light

        # Diffuse component calculation for directional lights
        cos_theta = max(np.dot(intersection.object.normal, light_direction), 0)
        diffuse_component = diffuse_color * light.color * cos_theta

        # Specular component calculation for directional lights
        light_direction_np = np.array(light_direction)
        intersection_normal_np = np.array(intersection.object.normal)

        reflection_direction = 2 * np.dot(light_direction_np,
                                          intersection_normal_np) * intersection_normal_np - light_direction_np

        cos_alpha = max(np.dot(reflection_direction, view_direction), 0)
        reduced_specular_color = specular_color * 0.2  # Adjust the multiplier as needed
        specular_component = reduced_specular_color * light.color * pow(cos_alpha, shininess)

        # Add diffuse and specular components for directional lights to final color
        final_color += diffuse_component + specular_component

    return tuple(map(int, np.clip(final_color, 0, 255)))  # Ensure color values are within range




def is_point_in_shadow(point, lights_point, lights_directional, objects, intersection):
    for obj in objects:
        if obj != intersection.object:  # Ignore the current object
            for light in lights_point + lights_directional:  # Consider both point and directional lights
                if light.type == 'point':
                    shadow_ray = Ray(point, np.subtract(light.position, point))
                elif light.type == 'directional':
                    shadow_ray = Ray(point, light.direction)  # Use light direction for directional light

                # Check the object type and call the appropriate method
                if isinstance(obj, Plane):
                    intersection_shadow = obj.intersect_shadow_ray(shadow_ray, light.position)
                elif isinstance(obj, Sphere):
                    intersection_shadow = obj.intersect_shadow_ray(shadow_ray, light.position)
                # Add similar checks for other object types here if necessary
                if intersection_shadow:
                    return True  # Point is in shadow
    return False  # Point is not in shadow


def parse_scene(scene_file):
    scene = None
    light_point = []
    light_directinal= []

    with open(scene_file, 'r') as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == 'image':
                scene = Scene(None, (int(tokens[1]), int(tokens[2])), None)
            elif tokens[0] == 'viewport':
                scene.viewport = list(map(float, tokens[1:]))
            elif tokens[0] == 'eye':
                scene.camera = list(map(float, tokens[1:4]))
            elif tokens[0] == 'sphere':
                # Parse sphere information: position, radius, color, material properties
                position = list(map(float, tokens[1:4]))
                radius = float(tokens[4])
                color = list(map(int, tokens[5:8]))
                material = {
                    'diffuse_color': list(map(int, tokens[8:11])),
                    'specular_color': list(map(int, tokens[11:14])),
                    'shininess': float(tokens[14])
                }
                sphere = Sphere(position, radius, color)
                scene.add_object(sphere, material)
            elif tokens[0] == 'plane':
                position = list(map(float, tokens[1:4]))
                normal = list(map(float, tokens[4:7]))
                color = list(map(int, tokens[7:10]))
                scene.add_object(Plane(position, normal, color))

            elif tokens[0] == 'light':
                light_type = tokens[1]
                if light_type == 'point':
                    position = list(map(float, tokens[2:5]))
                    direction = None
                    color = list(map(int, tokens[5:8]))
                    light_point.append(Light(light_type, position, direction, color))
                if light_type == 'directional':
                    position = None
                    direction = list(map(float, tokens[2:5]))
                    color = list(map(int, tokens[5:8]))
                    light_directinal.append(Light(light_type, position, direction, color))
                else:
                    continue


    return scene, light_point, light_directinal

def render_pixel(i, j, viewport, image_size, camera, objects, lights_point, lights_directional):
    x = viewport[0] + (viewport[2] - viewport[0]) * j / image_size[0]
    y = viewport[1] + (viewport[3] - viewport[1]) * i / image_size[1]
    ray = Ray(camera, [x, y, viewport[4]])

    closest_intersection = None
    for obj in objects:
        intersection = obj.intersect(ray)
        if intersection is not None and (closest_intersection is None or intersection.t < closest_intersection.t):
            closest_intersection = intersection

    if closest_intersection is not None:
        incident_direction = ray.direction
        normal_at_intersection = closest_intersection.object.normal
        view_direction = -incident_direction

        # Initialize colors for both types of lights
        pixel_color_point = (0, 0, 0)
        pixel_color_directional = (0, 0, 0)

        if lights_point:  # Check if there are point lights in the scene
            in_shadow_point = is_point_in_shadow(closest_intersection.point, lights_point, objects, closest_intersection)

            if not in_shadow_point:
                pixel_color_point = phong_lighting(closest_intersection, lights_point, view_direction,
                                                   closest_intersection.object.material, objects)

        if lights_directional:  # Check if there are directional lights in the scene
            in_shadow_directional = is_point_in_shadow(closest_intersection.point, lights_directional, objects, closest_intersection)

            if not in_shadow_directional:
                pixel_color_directional = phong_lighting(closest_intersection, lights_directional, view_direction,
                                                         closest_intersection.object.material, objects)

        # Combine the pixel colors from both types of lights (modify as needed)
        pixel_color = tuple(int(channel_point + channel_directional) for channel_point, channel_directional
                            in zip(pixel_color_point, pixel_color_directional))

        return (i, j, pixel_color)

    return (i, j, (0, 0, 0))





def render(scene, lights_point, lights_directional):
    image = np.zeros((scene.image_size[1], scene.image_size[0], 3), dtype=np.uint8)
    image_surface = pygame.display.set_mode(scene.image_size)

    # Create a thread pool with the number of workers equal to the number of CPU cores
    num_cores = multiprocessing.cpu_count()
    with Pool(2 * num_cores) as p:
        results = p.starmap(render_pixel,
                            [(i, j, scene.viewport, scene.image_size, scene.camera, scene.objects, lights_point, lights_directional)
                             for i in range(scene.image_size[1]) for j in range(scene.image_size[0])])

    # Set the pixel colors in the image based on the results
    for result in results:
        i, j, color = result
        pixel_color = tuple(map(int, color))  # Ensure color values are integers
        image[i, j] = pixel_color

    # Convert the image array to a Pygame surface
    pygame.surfarray.blit_array(image_surface, image)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    scene_file = "scene3.txt"  # Replace with the path to your file
    scene, lights_point, lights_directional = parse_scene(scene_file)
    # print(lights_point, lights_directional)

    print("Objects in the scene:")
    for obj in scene.objects:
        print(obj.__class__.__name__, "at", obj.position, "with color", obj.color, "and material", obj.material)

    # print(scene, lights_point, lights_directional)  # Verify the parsed scene details
    render(scene, lights_point, lights_directional)
