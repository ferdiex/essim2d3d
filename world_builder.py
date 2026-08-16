import pybullet as p
import pybullet_data

# Presets de fisica de contacto para obstaculos/paredes, con nombre en vez
# de numeros a memorizar. "grippy" (agarre) le sirve al GRU, que necesita
# friccion para pivotar/girar contra la pared. "slippery" (deslizamiento,
# los valores originales tipo pinball) le sirve a Braitenberg, cuyo
# mecanismo de escape (giro + kick de reversa) necesita desprenderse
# libremente, no agarrarse. Descubierto empiricamente, sesion 2026-08-07.
PHYSICS_PRESETS = {
    "grippy": {"lateralFriction": 0.35, "restitution": 0.25},
    "slippery": {"lateralFriction": 0.0, "restitution": 0.8},
}

def _resolve_physics(physics):
    """physics: nombre de preset ('grippy'/'slippery') o dict con
    lateralFriction/restitution propios."""
    if physics is None:
        physics = "grippy"
    if isinstance(physics, str):
        if physics not in PHYSICS_PRESETS:
            raise ValueError(f"Preset de fisica desconocido: {physics!r}. "
                              f"Opciones: {list(PHYSICS_PRESETS.keys())}")
        return PHYSICS_PRESETS[physics]
    return physics  # ya es un dict {lateralFriction, restitution}

def build_world(world_data, world_size=1.5, physics=None):
    """Devuelve la lista de body IDs creados (plano, paredes, bloques, meta),
    para poder limpiarlos despues con clear_world() sin reconectar PyBullet.
    Los llamadores que ya existian (essim3d.py) siguen funcionando igual,
    simplemente ignoran el valor de retorno.

    physics: 'grippy' (default), 'slippery', o un dict propio con
    lateralFriction/restitution."""
    phys = _resolve_physics(physics)
    ids = []
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    ids.append(p.loadURDF("plane.urdf"))
    ids.extend(build_basic_walls(world_size, physics=phys))

    if "rectangles" in world_data:
        for rect in world_data["rectangles"]:
            ids.append(spawn_block(rect["pos"], rect["size"], physics=phys))

    if "food_pos" in world_data:
        ids.extend(build_goal(world_data["food_pos"]))

    return ids

def clear_world(ids):
    """Borra los bodies creados por build_world, dejando la conexion de
    PyBullet intacta (robots incluidos, si no estan en la lista de ids)."""
    for body_id in ids:
        try:
            p.removeBody(body_id)
        except p.error:
            pass  # ya no existe (removido antes, o conexion cerrada)

def spawn_block(pos, size, physics=None):
    phys = _resolve_physics(physics)
    wall_height = 0.08
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, wall_height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, wall_height/2], rgbaColor=[0.420, 0.447, 0.502, 1.0])
    block_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[pos[0], pos[1], wall_height/2])
    p.changeDynamics(block_id, -1, lateralFriction=phys["lateralFriction"], restitution=phys["restitution"])
    return block_id

def build_basic_walls(world_size, physics=None):
    phys = _resolve_physics(physics)
    wall_thickness, wall_height = 0.02, 0.08 
    half_size = world_size / 2.0
    wall_color = [0.9, 0.314, 0.314, 1.0]
    
    walls = [
        ([0, half_size, wall_height/2], [world_size, wall_thickness, wall_height]),
        ([0, -half_size, wall_height/2], [world_size, wall_thickness, wall_height]),
        ([half_size, 0, wall_height/2], [wall_thickness, world_size, wall_height]),
        ([-half_size, 0, wall_height/2], [wall_thickness, world_size, wall_height])
    ]
    
    ids = []
    for p_pos, p_dim in walls:
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[d/2 for d in p_dim])
        v = p.createVisualShape(p.GEOM_BOX, halfExtents=[d/2 for d in p_dim], rgbaColor=wall_color)
        wall_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c, baseVisualShapeIndex=v, basePosition=p_pos)
        p.changeDynamics(wall_id, -1, lateralFriction=phys["lateralFriction"], restitution=phys["restitution"])
        ids.append(wall_id)
    return ids

def build_goal(pos):
    vis_base = p.createVisualShape(p.GEOM_CYLINDER, radius=0.1, length=0.001, rgbaColor=[1, 1, 0, 1])
    id_base = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_base, basePosition=[pos[0], pos[1], 0.001])
    vis_sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1.0, 0.98, 0.10, 0.80])
    id_sphere = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_sphere, basePosition=[pos[0], pos[1], 0.15])
    return [id_base, id_sphere]
