import numpy as np
from PIL import Image
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt

scene_file = "scene3.txt"
class Material:
    def __init__(self, diffuse_color, specular_color, shininess):
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.shininess = shininess

class Light:
    def __init__(self, light_type, color, position_or_direction):
        self.type = light_type
        self.color = color
        self.position_or_direction = position_or_direction

class Object:
    def __init__(self, material, position=None, normal=None):
        self.material = material
        self.position = position if position is not None else [0, 0, 0]
        self.normal = normal if normal is not None else [0, 1, 0]

class Sphere(Object):
    def __init__(self, center, radius, material, viewport=None, image_size=None, camera=None):
        super().__init__(material)
        self.center = center
        self.radius = radius
        self.viewport = viewport
        self.image_size = image_size
        self.camera = camera

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
    def __init__(self, position, normal, material, viewport=None, image_size=None, camera=None):
        super().__init__(material)
        self.position = position if position is not None else [0, -10, 0]  # default position is origin
        self.normal = normal if normal is not None else [0, 1, 0]  # default normal is upward
        self.viewport = viewport
        self.image_size = image_size
        self.camera = camera

    def intersect(self, ray):
        denom = np.dot(ray.direction, self.normal)
        if np.abs(denom) > 1e-6:
            t = np.dot(np.subtract(self.position, ray.origin), self.normal) / denom
            if t >= 0:
                point = np.add(ray.origin, np.multiply(t, ray.direction))
                return Intersection(t, point, self)  # Return the intersection point and the object
        return None

class Triangle(Object):
    def __init__(self, p1, p2, p3, material, viewport=None, image_size=None, camera=None):
        super().__init__(material)
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)
        self.viewport = viewport
        self.image_size = image_size
        self.camera = camera

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


class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)

class Intersection:
    def __init__(self, t, point, object):
        self.t = t
        self.point = point
        self.object = object

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

def parse_scene_file(scene_file):
    scene = None
    with open(scene_file, 'r') as file:
        for line in file:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == 'image':
                scene = Scene(None, (int(tokens[1]), int(tokens[2])), None)
            elif tokens[0] == 'viewport':
                scene.viewport = list(map(float, tokens[1:]))
            elif tokens[0] == 'eye':
                scene.camera = list(map(float, tokens[1:4]))
            elif tokens[0] == 'light':
                print("lights are here")
                light_type = tokens[1]
                color = list(map(int, tokens[2:5]))
                position_or_direction = list(map(float, tokens[5:]))
                scene.add_light(Light(light_type, color, position_or_direction))
            elif tokens[0] == 'sphere':
                material = Material(list(map(int, tokens[5:8])), list(map(int, tokens[8:11])), int(tokens[11]))
                scene.add_object(Sphere(list(map(float, tokens[1:4])), float(tokens[4]), material))
            elif tokens[0] == 'plane':
                if len(tokens) >= 17:  # Check if there are enough tokens to define a plane
                    material = Material(list(map(int, tokens[4:7])), list(map(int, tokens[7:10])), int(tokens[10]))
                    scene.add_object(Plane(list(map(float, tokens[1:4])), list(map(float, tokens[4:7])), material))
                else:
                    print("Not enough information for defining a plane.")
            elif tokens[0] == 'tri':
                if len(tokens) >= 17:  # Check if there are enough tokens to define a triangle
                    material = Material(list(map(int, tokens[10:13])), list(map(int, tokens[13:16])), int(tokens[16]))
                    p1 = list(map(float, tokens[1:4]))
                    p2 = list(map(float, tokens[4:7]))
                    p3 = list(map(float, tokens[7:10]))
                    scene.add_object(Triangle(p1, p2, p3, material))
                else:
                    print("Not enough information for defining a triangle.")
    return scene



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

    Image.fromarray(image).save('output.png')
    plt.imshow(image)
    plt.show()

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
        # Pass all lights in the scene to the shading function
        return (i, j, phong_shading(closest_intersection.object.material, scene.lights, closest_intersection))
    return (i, j, [0, 0, 0])  # Background color if no intersection



def phong_shading(material, lights, intersection):
    ambient_color = np.array([10, 10, 10])  # Some ambient color, modify as needed
    object_color = np.array(material.diffuse_color)
    final_color = np.zeros(3)

    for light in lights:
        # Calculate lighting contributions for each light
        light_direction = np.array(light.position_or_direction) - np.array(intersection.point)
        light_distance = np.linalg.norm(light_direction)
        light_direction /= light_distance

        # Ensure the normal vector is normalized
        object_normal = intersection.object.normal / np.linalg.norm(intersection.object.normal)

        # Calculate diffuse component
        diffuse_intensity = max(0, np.dot(object_normal, light_direction))
        diffuse = material.diffuse_color * light.color * diffuse_intensity

        # Calculate specular component
        reflection = 2 * np.dot(object_normal, light_direction) * object_normal - light_direction
        view_direction = -np.array(intersection.point)
        specular_intensity = max(0, np.dot(reflection, view_direction) ** material.shininess)
        specular = material.specular_color * light.color * specular_intensity

        # Accumulate the contributions from all lights
        final_color += ambient_color + diffuse + specular

    final_color = np.clip(final_color, 0, 255)  # Ensure color values are within [0, 255]
    return final_color.astype(int).tolist()


    # # Example of applying transformation to objects
    # transformation = Transform()
    # transformation.translate(1, 2, 3)  # Example translation
    #
    # # Apply the transformation to an object (replace 'obj' with the actual object)
    # obj.transform(transformation)


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
        selected_parent.color = [255, 255, 0]  # Set the color to yellow
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

def main():
    scene = parse_scene_file(scene_file)
    scene_hierarchy = SceneHierarchy()
    selected_parent = None  # Initialize selected_parent outside the loop

    original_lights = scene.lights

    main_scene = Scene(scene.viewport, scene.image_size, scene.camera)
    render(main_scene)

    while True:
        if selected_parent is None:  # Allow selecting the parent only if it's not selected
            print("Select a parent object:")
            selected_parent = select_parent_object(scene)
            if selected_parent:
                scene_hierarchy.set_parent(selected_parent)
                for obj in scene.objects:
                    if obj != selected_parent:
                        scene_hierarchy.add_child(obj)
                # Add the parent object and its children to the main scene
                main_scene.add_object(selected_parent)
                for child in scene_hierarchy.children:
                    main_scene.add_object(child)

        print("\n1. Render the scene")
        print("2. Apply transformation to objects")
        print("3. Exit")

        choice = int(input("Enter your choice: "))

        if choice == 1:
            render(main_scene)
        elif choice == 2:
            if selected_parent:
                scene_hierarchy.apply_transform()
            else:
                print("Please select a parent object first.")
        elif choice == 3:
            break
        else:
            print("Invalid choice. Please enter a valid option.")


if __name__ == "__main__":
    main()
