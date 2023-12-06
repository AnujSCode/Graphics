#!/usr/bin/env python3

import numpy as np
from PIL import Image
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pygame

scene_file = "scene4.txt"


class Light:
    def __init__(self, light_type, position, direction, color):
        self.type = light_type
        self.position = position if light_type == 'point' else None
        self.direction = direction if light_type == 'directional' else None
        self.color = color

        if light_type == 'directional' and direction is not None:
            # Convert the provided direction into a normalized vector
            direction = np.array(direction)  # Convert to NumPy array for vector operations
            magnitude = np.linalg.norm(direction)  # Calculate magnitude
            self.direction = direction / magnitude  # Normalize the direction vector


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


def render_pixel(i, j, viewport, image_size, camera, objects, lights):
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
        view_direction = -incident_direction

        pixel_color = np.zeros(3)  # Initialize color as black

        for light in lights:
            shadow_ray = None
            if light.type == 'point':
                shadow_ray_direction = normalize(light.position - closest_intersection.point)
                shadow_ray_origin = closest_intersection.point +.001 * shadow_ray_direction # Offset origin slightly
                shadow_ray = Ray(shadow_ray_origin, shadow_ray_direction)
            elif light.type == 'directional':
                shadow_ray_direction = normalize(-light.direction)
                shadow_ray_origin = closest_intersection.point + 0.001 * shadow_ray_direction  # Offset origin slightly
                shadow_ray = Ray(shadow_ray_origin, shadow_ray_direction)

            if shadow_ray:
                in_shadow = False
                for obj in objects:
                    if obj != closest_intersection.object and obj.intersect(shadow_ray):
                        in_shadow = True
                        break

                if in_shadow:
                    pixel_color = phong_model(closest_intersection, light, view_direction,
                                              ambient_intensity=0.1, light_intensity=1, shininess=50)
                    shadow_color = tuple(int(c * 0.6) for c in pixel_color)
                    pixel_color = shadow_color
                else:
                    pixel_color += phong_model(closest_intersection, light, view_direction,
                                               ambient_intensity=0.1, light_intensity=0.8, shininess=50)

        return (i, j, tuple(int(c) for c in pixel_color))

    return (i, j, (0, 0, 0))  # Return black for no intersection


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
        # Assuming pixel_color is an out-of-bound Python integer being assigned to an image array
        image[i, j] = np.array(pixel_color).astype(image.dtype)

    # Convert the image array to a Pygame surface
    pygame.surfarray.blit_array(image_surface, image)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()

    pygame.quit()


def phong_model(intersection, light, viewer, ambient_intensity, light_intensity, shininess):
    # Calculate vectors
    normal = intersection.object.normal_at(intersection.point)

    if light.type == 'point':
        light_dir = normalize(light.position - intersection.point)
    elif light.type == 'directional':

        light_dir = normalize(-light.direction)  # Reverse direction for directional light

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
        self.default_color = color

    def change_color(self, new_color):
        self.color = new_color  # Change the color attribute to the new color

    def reset_color(self):
        self.color = self.default_color  # Reset the color back to the default color

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

    def normal_at(self, point):
        # Calculate the normalized normal vector at a given point on the sphere
        return (point - self.center) / np.linalg.norm(point - self.center)

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
        self.default_color = color

    def change_color(self, new_color):
        self.color = new_color  # Change the color attribute to the new color

    def reset_color(self):
        self.color = self.default_color  # Reset the color back to the default color

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

    def normal_at(self, point):
        # For a plane, the normal is constant
        return self.normal

    def transform(self, transformation):
        # Transform the plane by applying the transformation matrix to its position
        position_homogeneous = np.append(self.position, 1)  # Convert to homogeneous coordinates
        transformed_position_homogeneous = np.dot(transformation.get_matrix(), position_homogeneous)
        self.position = transformed_position_homogeneous[:3]  # Convert back to 3D coordinates

        # Transform the normal vector of the plane
        normal_matrix = np.linalg.inv(np.transpose(transformation.get_matrix()))
        normal_homogeneous = np.append(self.normal, 0)  # Convert to homogeneous coordinates
        transformed_normal_homogeneous = np.dot(normal_matrix, normal_homogeneous)
        self.normal = transformed_normal_homogeneous[:3]  # Convert back to 3D coordinates and normalize

        # Normalize the transformed normal
        self.normal /= np.linalg.norm(self.normal)

class Triangle(Object):
    def __init__(self, v0, v1, v2, color):
        super().__init__(color)
        self.v0 = np.array(v0)
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.default_color = color

    def change_color(self, new_color):
        self.color = new_color  # Change the color attribute to the new color

    def reset_color(self):
        self.color = self.default_color  # Reset the color back to the default color

    def normal_at(self, point):
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        return np.cross(edge1, edge2) / np.linalg.norm(np.cross(edge1, edge2))

    def intersect(self, ray):
        epsilon = 1e-6

        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        h = np.cross(ray.direction, edge2)
        a = np.dot(edge1, h)

        if -epsilon < a < epsilon:
            return None  # Ray is parallel to the triangle

        f = 1.0 / a
        s = ray.origin - self.v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return None

        q = np.cross(s, edge1)
        v = f * np.dot(ray.direction, q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * np.dot(edge2, q)
        if t > epsilon:
            intersection_point = ray.origin + t * ray.direction
            return Intersection(t, intersection_point, self)

        return None  # No intersection within valid bounds

    def intersect_shadow_ray(self, shadow_ray, light_direction):
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0

        pvec = np.cross(shadow_ray.direction, edge2)
        det = np.dot(edge1, pvec)

        if det > -0.000001 and det < 0.000001:
            return False
        inv_det = 1.0 / det

        tvec = shadow_ray.origin - self.v0
        u = np.dot(tvec, pvec) * inv_det

        if u < 0.0 or u > 1.0:
            return False

        qvec = np.cross(tvec, edge1)
        v = np.dot(shadow_ray.direction, qvec) * inv_det

        if v < 0.0 or u + v > 1.0:
            return False

        t = np.dot(edge2, qvec) * inv_det

        return t > 0.000001

    def transform(self, transformation):
        # Transform each vertex of the triangle by applying the transformation matrix
        v0_homogeneous = np.append(self.v0, 1)
        transformed_v0_homogeneous = np.dot(transformation.get_matrix(), v0_homogeneous)
        self.v0 = transformed_v0_homogeneous[:3]

        v1_homogeneous = np.append(self.v1, 1)
        transformed_v1_homogeneous = np.dot(transformation.get_matrix(), v1_homogeneous)
        self.v1 = transformed_v1_homogeneous[:3]

        v2_homogeneous = np.append(self.v2, 1)
        transformed_v2_homogeneous = np.dot(transformation.get_matrix(), v2_homogeneous)
        self.v2 = transformed_v2_homogeneous[:3]

class Scene:
    def __init__(self, viewport, image_size, camera):
        self.viewport = viewport
        self.image_size = image_size
        self.camera = camera
        self.objects = []
        self.floor = []
        self.lights = []

    def add_object(self, obj, material=None):
        if material is not None:
            obj.material = material
        self.objects.append(obj)

    def add_floor(self, obj, material=None):
        if material is not None:
            obj.material = material
        self.objects.append(obj)

    # Add a method to add lights to the scene
    def add_light(self, light):
        self.lights.append(light)


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
                material = {
                    'diffuse_color': list(map(int, tokens[10:13])),
                    'specular_color': list(map(int, tokens[13:16])),
                    'shininess': float(tokens[16])
                }
                scene.add_object(Plane(position, normal, color), material)
            elif tokens[0] == 'triangle':
                v0 = list(map(float, tokens[1:4]))
                v1 = list(map(float, tokens[4:7]))
                v2 = list(map(float, tokens[7:10]))
                color = list(map(int, tokens[10:13]))
                material = {
                    'diffuse_color': list(map(int, tokens[13:16])),
                    'specular_color': list(map(int, tokens[16:19])),
                    'shininess': float(tokens[19])
                }
                triangle = Triangle(v0, v1, v2, color)
                scene.add_object(triangle, material)
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

    def add_child_without_transform(self, child_object):
        # Add the child without applying transformations
        self.children.append(child_object)

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

        # Update the color of the selected object to orange
        selected_parent.change_color([255, 165, 0])  # Orange color in RGB

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
    scene_file = "scene4.txt"
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
    for obj in scene.objects:
        if obj != selected_parent:
            if isinstance(obj, Plane):  # Check if the object is a plane
                main_scene.add_floor(obj)  # Add the plane directly to the main scene
            else:
                main_scene.add_object(obj)  # Add other objects as children

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
