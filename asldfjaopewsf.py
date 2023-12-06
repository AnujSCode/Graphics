from everything import Scene, Ray, Intersection, Triangle, render, Light, render_pixel

def  read_scene_file(scene_file):
    triangles = []
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
            elif tokens[0] == 'triangle':
                v0 = list(map(float, tokens[1:4]))
                v1 = list(map(float, tokens[4:7]))
                v2 = list(map(float, tokens[7:10]))
                color = list(map(int, tokens[10:13]))
                print(f"Parsed triangle: v0={v0}, v1={v1}, v2={v2}, color={color}")
                triangles.append(Triangle(v0, v1, v2, color))
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
    if scene is not None:
        for triangle in triangles:
            scene.add_object(triangle)
    return scene, lights
if __name__ == "__main__":
    scene_file = "scene4.txt"  # Replace with your scene file
    scene, lights = read_scene_file(scene_file)

    for obj in scene.objects:
        scene.add_object(obj)

    print(f"Number of triangles in scene: {len(scene.objects)}")

