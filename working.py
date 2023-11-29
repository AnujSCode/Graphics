#!/usr/bin/env python3

import numpy as np
from PIL import Image
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pygame


scene_file = "scene3.txt"

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


def intersect_scene(ray, objects):
    closest_intersection = None

    for obj in objects:
        intersection = obj.intersect(ray)
        if intersection is not None:
            if closest_intersection is None or intersection.t < closest_intersection.t:
                closest_intersection = intersection

    return closest_intersection

def phong_lighting(intersection, lights, view_direction, material, objects):
    ambient_color = np.array([20, 20, 20])  # Ambient color
    light_direction = None
    light_distance = None


    # Default values if material properties are missing
    default_diffuse_color = np.array([225, 225, 225])
    default_specular_color = np.array([255, 255, 255])
    default_shininess = 50

    # Extract material properties if available, otherwise, use default values
    diffuse_color = np.array(material.get('diffuse_color', default_diffuse_color))
    specular_color = np.array(material.get('specular_color', default_specular_color))
    shininess = material.get('shininess', default_shininess)

    object_color = np.array(intersection.object.color)

    total_color = ambient_color * object_color # Initialize with ambient light

    # Debugging lines for ambient light


    for light in lights:
        print(light)
        light_intensity = np.array(light.color) / 255.0

        if light.type == 'point':
            light_direction = np.array(light.position) - np.array(intersection.point)
            light_distance = np.linalg.norm(light_direction)
            light_direction = light_direction / light_distance  # Normalize light direction
            print(f"Point Light Intensity: {light_intensity}")
            print(f"Light Direction: {light_direction}")
        elif light.type == 'directional':
            light_direction = np.array(light.direction)
            light_distance = float('inf')  # Consider directional light at infinity
            print(f"Directional Light Intensity: {light_intensity}")
            print(f"Light Direction: {light_direction}")

        # Diffuse component
        diffuse_factor = max(np.dot(intersection.object.normal, light_direction), 0)
        diffuse_term = diffuse_color * light_intensity * diffuse_factor

        # Specular component
        reflection = 2 * np.dot(intersection.object.normal, light_direction) * np.array(intersection.object.normal) - light_direction
        reflection /= np.linalg.norm(reflection)  # Normalize the reflection vector
        specular_factor = max(np.dot(reflection, view_direction), 0) ** shininess  # Considering shininess
        print(f"Specular Factor: {specular_factor}")

        specular_term = specular_color * light_intensity * specular_factor

        # Print out intermediate values for debugging
        print("Diffuse Term:", diffuse_term)
        print("Specular Term:", specular_term)

        # Shadows: Check if point is in shadow
        shadow_ray = Ray(intersection.point + 0.001 * light_direction, light_direction)
        shadow_intersection = intersect_scene(shadow_ray, objects)  # Provide 'objects' argument

        if shadow_intersection is not None and np.linalg.norm(
                shadow_intersection.point - intersection.point) < light_distance:
            # Point is in shadow, don't add diffuse and specular terms
            total_color += ambient_color * object_color
        else:
            total_color += (ambient_color * object_color + diffuse_term + specular_term).astype(np.uint8)


    total_color = total_color.astype(np.uint8)
    # Clip each color channel individually and return as a tuple of integers
    clipped_color = tuple(np.clip(total_color.astype(int), 0, 255))

    return clipped_color



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


class Plane(Object):
    def __init__(self, position, normal, color):
        super().__init__(color)
        self.position = position if position is not None else [0, 0, 0]  # default position is origin
        self.normal = normal if normal is not None else [0, 1, 0]  # default normal is upward

    def intersect(self, ray):
        denom = np.dot(ray.direction, self.normal)
        if np.abs(denom) > 1e-6:
            t = np.dot(np.subtract(self.position, ray.origin), self.normal) / denom
            if t >= 0:
                point = np.add(ray.origin, np.multiply(t, ray.direction))
                return Intersection(t, point, self)  # Return the intersection point and the object
        return None

class Triangle(Object):
    def __init__(self, p1, p2, p3, color):
        super().__init__(color)
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

# Your other code...

def render_pixel(i, j, viewport, image_size, camera, objects, lights):
    for light in lights:
        print(light)
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

        if lights:  # Check if there are lights in the scene
            in_shadow = is_point_in_shadow(closest_intersection.point, lights, objects, closest_intersection)

            if in_shadow:
                pixel_color = (50, 50, 50)
            else:
                pixel_color = phong_lighting(closest_intersection, lights, view_direction, closest_intersection.object.material, objects)
        else:
            pixel_color = phong_lighting(closest_intersection, [], view_direction, closest_intersection.object.material, objects)

        return (i, j, pixel_color)
    return (i, j, (0, 0, 0))





def render(scene, lights):
    image = np.zeros((scene.image_size[1], scene.image_size[0], 3), dtype=np.uint8)
    image_surface = pygame.display.set_mode(scene.image_size)

    # Create a thread pool with the number of workers equal to the number of CPU cores
    num_cores = multiprocessing.cpu_count()
    with Pool(2 * num_cores) as p:
        results = p.starmap(render_pixel,
                            [(i, j, scene.viewport, scene.image_size, scene.camera, scene.objects, lights)
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


def is_point_in_shadow(point, lights, objects, closest_intersection):
    in_shadow = False

    for light in lights:
        if light.position is None:
            continue  # Skip this light if position is None

        light_direction = np.array(light.position) - point

        if closest_intersection.point is None:
            continue  # Skip further calculations if closest_intersection.point is None

        distance_to_light = np.linalg.norm(light_direction)
        light_direction /= distance_to_light  # Normalize light direction

        shadow_ray = Ray(point + 0.001 * light_direction, light_direction)

        shadow_hit_count = 0
        for obj in objects:
            intersection = obj.intersect(shadow_ray)
            if intersection is not None and np.linalg.norm(intersection.point - point) < distance_to_light:
                shadow_hit_count += 1  # Increment shadow hit count
                # You might consider breaking the loop here or accumulating hits for soft shadows

        # Adjust the threshold for shadow detection
        if shadow_hit_count > 0.5 * len(objects):  # Adjust threshold as needed
            in_shadow = True  # Point is in shadow

        if in_shadow:
            break  # No need to check further, in shadow for any light

    return in_shadow



# Update read_scene_file to handle lights and material properties
def read_scene_file(scene_file):
    scene = None
    lights = []

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
                elif light_type == 'directional':
                    position = None
                    direction = list(map(float, tokens[2:5]))
                else:
                    continue

                color = list(map(int, tokens[5:8]))
                lights.append(Light(light_type, position, direction, color))
    return scene, lights

class SceneHierarchy:
    def __init__(self):
        self.parent = None
        self.children = []

    def set_parent(self, parent_object):
        self.parent = parent_object

    def add_child(self, child_object):
        if self.parent is not None:
            self.children.append(child_object)
        else:
            print("Error: Set a parent object before adding children.")

    def render(self):
        if self.parent is not None:
            scene = Scene(self.parent.viewport, self.parent.image_size, self.parent.camera)
            scene.add_object(self.parent)

            for child in self.children:
                scene.add_object(child)

            render(scene)
        else:
            print("Error: No parent object specified.")

    def apply_transform(self):
        if self.parent is not None:
            # Create a Transform object
            transformation = Transform()

            # Prompt the user for the type of transformation
            print("Choose a transformation:")
            print("1. Translate")
            print("2. Rotate (X-axis)")
            print("3. Rotate (Y-axis)")
            print("4. Rotate (Z-axis)")
            print("5. Scale")
            choice = int(input("Enter your choice: "))

            if choice == 1:  # Translate
                x = float(input("Enter the translation along the X-axis: "))
                y = float(input("Enter the translation along the Y-axis: "))
                z = float(input("Enter the translation along the Z-axis: "))
                transformation.translate(x, y, z)
            elif choice == 2:  # Rotate (X-axis)
                angle = float(input("Enter the rotation angle (degrees): "))
                transformation.rotate_x(angle)
            elif choice == 3:  # Rotate (Y-axis)
                angle = float(input("Enter the rotation angle (degrees): "))
                transformation.rotate_y(angle)
            elif choice == 4:  # Rotate (Z-axis)
                angle = float(input("Enter the rotation angle (degrees): "))
                transformation.rotate_z(angle)
            elif choice == 5:  # Scale
                sx = float(input("Enter the scaling factor along the X-axis: "))
                sy = float(input("Enter the scaling factor along the Y-axis: "))
                sz = float(input("Enter the scaling factor along the Z-axis: "))
                transformation.scale(sx, sy, sz)
            else:
                print("Invalid choice. No transformation applied.")

            # Apply the transformation to the parent object
            self.parent.transform(transformation)

            # Apply the same transformation to all the children
            for child in self.children:
                child.transform(transformation)
        else:
            print("Error: No parent object specified.")


def select_parent_object(scene):
    print("Select a parent object from the following objects:")
    for idx, obj in enumerate(scene.objects):
        print(f"{idx}: {obj.__class__.__name__}")

    parent_idx = int(input("Enter the index of the parent object: "))

    if parent_idx >= 0 and parent_idx < len(scene.objects):
        selected_parent = scene.objects[parent_idx]

        # Update material properties for shininess
        selected_parent.material = {
            'diffuse_color': [255, 255, 255],  # Adjust diffuse and specular colors as needed
            'specular_color': [255, 255, 255],
            'shininess': 50  # Set the shininess value as desired
        }

        return selected_parent
    else:
        print("Invalid index. Please select a valid parent object.")
        return None


class Transform:
    def __init__(self):
        # Initialize the transformation matrix as the identity matrix (no transformation)
        self.matrix = np.identity(4, dtype=float)

    def translate(self, x, y, z):
        # Apply a translation to the transformation matrix
        translation_matrix = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=float)
        self.matrix = np.dot(self.matrix, translation_matrix)

    def rotate_x(self, angle_degrees):
        # Apply a rotation around the X-axis to the transformation matrix
        angle_radians = np.radians(angle_degrees)
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
            [0, np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ], dtype=float)
        self.matrix = np.dot(self.matrix, rotation_matrix)

    def rotate_y(self, angle_degrees):
        # Apply a rotation around the Y-axis to the transformation matrix
        angle_radians = np.radians(angle_degrees)
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ], dtype=float)
        self.matrix = np.dot(self.matrix, rotation_matrix)

    def rotate_z(self, angle_degrees):
        # Apply a rotation around the Z-axis to the transformation matrix
        angle_radians = np.radians(angle_degrees)
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        self.matrix = np.dot(self.matrix, rotation_matrix)

    def scale(self, sx, sy, sz):
        # Apply a scaling to the transformation matrix
        scale_matrix = np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        self.matrix = np.dot(self.matrix, scale_matrix)

    def get_matrix(self):
        return self.matrix

if __name__ == "__main__":
    scene_file = "scene3.txt"
    selected_input = ""
    # Read the scene file and create the initial scene object
    scene, lights = read_scene_file(scene_file)
    # Create a new Scene object for the entire scene
    main_scene = Scene(scene.viewport, scene.image_size, scene.camera)

    render(scene, lights)
    # Select the parent object interactively
    selected_parent = None

    while selected_parent is None:
        selected_parent = select_parent_object(scene)

    # Create a SceneHierarchy object
    scene_hierarchy = SceneHierarchy()

    # Set the selected parent object
    scene_hierarchy.set_parent(selected_parent)

    # Add other objects as children
    for obj in scene.objects:
        if obj != selected_parent:
            scene_hierarchy.add_child(obj)

    # Add the parent object and its children to the main scene
    main_scene.add_object(selected_parent)
    for child in scene_hierarchy.children:
        if isinstance(child, Sphere):
            main_scene.add_object(child)


    # Initial render before user interaction
    render(main_scene, lights)

    while selected_input != "exit":
        # Add the parent object and its children to the main scene
        main_scene.add_object(selected_parent)
        for child in scene_hierarchy.children:
            main_scene.add_object(child)
        for light in lights:
            main_scene.add_light(light)

            # Apply transformations to the parent object
        scene_hierarchy.apply_transform()

        # Render the entire hierarchy
        render(main_scene, lights)