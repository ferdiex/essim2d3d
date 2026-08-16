import math
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated*",
    category=UserWarning
)
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from numba import njit
from config_loader import load_robot_config, load_world_file

@njit
def fast_raycast(pos, direction, obstacles, world_width, world_height, sensor_range):
    for t in range(1, int(sensor_range) + 1):
        px = pos[0] + direction[0] * t
        py = pos[1] + direction[1] * t
        if not (0 <= px <= world_width and 0 <= py <= world_height): return float(t)
        for ox, oy, ow, oh in obstacles:
            if ox <= px <= ox + ow and oy <= py <= oy + oh: return float(t)
    return float(sensor_range)

@dataclass
class ForagingEnvConfig:
    world_width: int = 800
    world_height: int = 600
    robot_radius: float = 12.0
    collision_radius: float = 12.0 
    food_radius: float = 65.0        
    success_threshold: float = 65.0 
    odor_radius: float = 5.0
    sensor_range: float = 110.0
    num_sensors: int = 8
    max_steps: int = 600            
    num_agents: int = 2             
    comm_decay: float = 200.0       
    render_fps: int = 30
    map_name: str = "default"
    agent_colors: List[Tuple] = None
    u_env: bool = False

    # NUEVO
    randomize_spawns: bool = False
    randomize_orientations: bool = True
    randomize_food: bool = False
    social_mode: str = "normal"   # normal | off | shuffled | mute_sender
    social_variant: str = "angle_strength"   # angle_strength | angle_only | strength_only

    def __post_init__(self):
        if self.agent_colors is None:
            self.agent_colors = [(40, 120, 220), (40, 200, 200), (200, 40, 200)]

class ForagingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Optional[ForagingEnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config or ForagingEnvConfig()
        self.render_mode = render_mode
        if self.config.num_agents > 1:
            self.action_space = spaces.Discrete(5) 
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.config.num_sensors + 3,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4) 
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.config.num_sensors + 2,), dtype=np.float32)
        
        self.agent_pos = np.zeros((self.config.num_agents, 2), dtype=np.float32)
        self.agent_theta = np.zeros(self.config.num_agents, dtype=np.float32)
        self.agent_signals = np.zeros(self.config.num_agents, dtype=np.float32)
        self.agent_success = [False] * self.config.num_agents
        self.food_pos = np.zeros(2, dtype=np.float32)
        self.step_count = 0        
        self._build_map()
        self.window, self.clock = None, None
        self.agent_success_step = [-1] * self.config.num_agents
        self.signal_counts = np.zeros(self.config.num_agents, dtype=np.int32)
        
        # Initialize trajectory history
        self.agent_paths = [[] for _ in range(self.config.num_agents)] #Live Path Tracing

    def _m_to_px(self, m_pos, m_size):
        """Converts coordinates from meters (center 0,0) to pixels (Pygame)."""
        scale = 800 / 1.5  # 800 pixeles / 1.5 metros
        offset = 400       # Centro de la pantalla
        
        # El JSON usa el CENTRO del bloque, Pygame usa la ESQUINA superior izquierda
        px_x = (m_pos[0] * scale) + offset - (m_size[0] * scale / 2)
        px_y = (m_pos[1] * scale) + offset - (m_size[1] * scale / 2)
        
        width = m_size[0] * scale
        height = m_size[1] * scale
        
        return (int(px_x), int(px_y), int(width), int(height))

    def _build_map(self):
        """Loads the world from JSON and translates it into pixels (3D-aligned)."""
        # 1. Determinar qué archivo cargar (Mapeo de nombres)
        # Si el usuario pone --map default, cargamos el de los 5 bloques (random_obstacles)
        if self.config.map_name == "default":
            world_name = "random_obstacles"
        else:
            # If you specify `--map u_shape` or `n_shape`, we use that name directly
            world_name = self.config.map_name
        
        # 2. Load the JSON file from the /worlds folder.
        world_data = load_world_file(world_name)
        
        if world_data and "rectangles" in world_data:
            self.obstacles = []
            scale = 800 / 1.5
            
            for rect in world_data["rectangles"]:
                
                # We convert each rectangle from meters to pixels using the previous logic
                px_rect = self._m_to_px(rect["pos"], rect["size"])
                self.obstacles.append(px_rect)
            
            # 3. SYNCHRONIZING FOOD: We maintain the scale logic
            f_pos = world_data.get("food_pos", [0, 0])
            self.food_pos = np.array([
                (f_pos[0] * scale) + 400,
                (f_pos[1] * scale) + 400
            ], dtype=np.float32)
            
            # Success log
            # print(f"[SUCCESS] Mapa '{world_name}' cargado con {len(self.obstacles)} bloques.")
        else:
            # Failsafe: If the file does not exist, we don't crash the program.a
            print(f"[ERROR] The worlds file could not be found/{world_name}.json")
            self.obstacles = []
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.agent_success = [False] * self.config.num_agents
        self.agent_signals.fill(0.0)
        self.agent_success_step = [-1] * self.config.num_agents
        self.signal_counts.fill(0)

        # --- 1. MOVE BLOCKS FIRST (SO THEY DON'T STEP ON THE ROBOT) ---
        if not self.config.u_env and self.config.map_name not in ["u_shape", "n_shape"]:
            jitter = 40 
            self.obstacles = [
                (150 + np.random.uniform(-jitter, jitter), 100 + np.random.uniform(-jitter, jitter), 60, 150),
                (550 + np.random.uniform(-jitter, jitter), 100 + np.random.uniform(-jitter, jitter), 60, 150),
                (350 + np.random.uniform(-jitter, jitter), 250 + np.random.uniform(-jitter, jitter), 100, 60),
                (150 + np.random.uniform(-jitter, jitter), 400 + np.random.uniform(-jitter, jitter), 120, 40),
                (550 + np.random.uniform(-jitter, jitter), 400 + np.random.uniform(-jitter, jitter), 120, 40)
            ]

        # --- 2. FOOD(SECOND) ---
        if self.config.randomize_food:
            if self.config.u_env:
                self.food_pos = np.array([np.random.uniform(280, 500), np.random.uniform(280, 430)], dtype=np.float32)
            else:
                self.food_pos = self._sample_food().astype(np.float32)
        else:
            self.food_pos = np.array([400, 350], dtype=np.float32) if self.config.u_env else np.array([400, 100], dtype=np.float32)

        # --- 3. AGENTS (AT THE END, ONCE EVERYONE IS READY) ---
        for i in range(self.config.num_agents):
            p, th = self._sample_pose(agent_idx=i, exclude_agents=list(range(i)))
            self.agent_pos[i] = p
            self.agent_theta[i] = th
        
        # --- 4. Clear traces on restart - Live Path Tracing
        self.agent_paths = [[] for _ in range(self.config.num_agents)]

        return self._get_obs_all(), self._get_info()

    def step(self, actions: List[int]):
        self.step_count += 1

        for i in range(len(actions)):
            # Signal Record
            if self.config.num_agents > 1:
                if self.config.social_mode == "mute_sender":
                    self.agent_signals[i] = 0.0
                else:
                    self.agent_signals[i] = 1.0 if actions[i] == 4 else 0.0
                    if self.agent_signals[i] > 0.5:
                        self.signal_counts[i] += 1

            # Movement / success
            if not self.agent_success[i]:
                self._apply_action(i, actions[i])
                dist = np.linalg.norm(self.agent_pos[i] - self.food_pos)
                if dist < self.config.success_threshold:
                    self.agent_success[i] = True
                    if self.agent_success_step[i] < 0:
                        self.agent_success_step[i] = self.step_count
                        
            if not self.agent_success[i]: # Live Path Tracing
                # Save the current position to the trail
                self.agent_paths[i].append((float(self.agent_pos[i][0]), float(self.agent_pos[i][1])))

        return (
            self._get_obs_all(),
            0.0,
            all(self.agent_success),
            self.step_count >= self.config.max_steps,
            self._get_info(),
        )
    
    def _sample_pose(self, agent_idx=0, exclude_agents=[]):
        for _ in range(1000):
            # Agente 0 (Líder): Spawnea en la mitad superior (más cerca de meta)
            # Agente 1 (Seguidor): Spawnea en la mitad inferior (castigado)
            if agent_idx == 0:
                p = np.array([np.random.uniform(100, 700), np.random.uniform(40, 300)], dtype=np.float32)
            else:
                p = np.array([np.random.uniform(40, 760), np.random.uniform(400, 560)], dtype=np.float32)
            
            if self._collides(-1, p): continue
            
            # ASYMMETRY RULE: The follower (1) must be far from the food (> 400px)
            dist_to_food = np.linalg.norm(p - self.food_pos)
            
            if dist_to_food < 100.0: continue 

            if agent_idx == 1 and dist_to_food < 400.0: continue
            if agent_idx == 0 and dist_to_food > 300.0: continue # Líder cerca

            # U-lock
            if self.config.u_env and (310 < p[0] < 510 and 240 < p[1] < 470): continue 
            
            # Avoid Colissions
            too_close = any(np.linalg.norm(p - self.agent_pos[idx]) < 50 for idx in exclude_agents)
            
            if not too_close:
                # MOD 2: Biased Orientation 
                # The leader (0) looks ahead (towards the finish line). The follower (1) looks back (lost).
                if agent_idx == 0:
                    theta = float(np.random.uniform(-math.pi/2, math.pi/2)) # Look Up
                else:
                    theta = float(np.random.uniform(math.pi/2, 3*math.pi/2)) # Look Down
                
                return p, theta
        
        return np.array([100, 100]), 0.0

    def _sample_food(self):
        for _ in range(1000):
            p = np.array([np.random.uniform(100, 700), np.random.uniform(100, 500)], dtype=np.float32)

            # We verify that the food circle does not touch any wall
            # We use a margin of 50 to ensure there is clearance
            collision = False
            for ox, oy, ow, oh in self.obstacles:
                nx, ny = np.clip(p[0], ox, ox+ow), np.clip(p[1], oy, oy+oh)
                if np.linalg.norm(p - [nx, ny]) < 50.0: # Margen de seguridad
                    collision = True
                    break
            if not collision: return p
        return np.array([400, 300])

    def _apply_action(self, i, action): 
            pos, theta = self.agent_pos[i].copy(), self.agent_theta[i]
        
            # --- Action 0 and Action 4 move the robot --
            if action == 0 or action == 4: 
                pos += np.array([math.cos(theta), math.sin(theta)]) * 8.0
            elif action == 1: theta -= math.radians(20)
            elif action == 2: theta += math.radians(20)
            elif action == 3: pos -= np.array([math.cos(theta), math.sin(theta)]) * 6.0
        
            if not self._collides(i, pos): self.agent_pos[i] = pos
            self.agent_theta[i] = self._wrap(theta)

    def _get_obs_single(self, i: int) -> np.ndarray:
        # 1. Proximity with Realistic Noise (V50)
        prox = []
        angle_step = 2.0 * math.pi / self.config.num_sensors
        for j in range(self.config.num_sensors):
            angle = self.agent_theta[i] + j * angle_step
            dist = self._raycast(i, angle)
            val = float(np.clip(1.0 - dist / self.config.sensor_range, 0.0, 1.0))
            # 3% Noise Injection
            prox.append(float(np.clip(val + np.random.normal(0, 0.03), 0.0, 1.0)))
            
        # 2. Food Variables
        dx, dy = self.food_pos - self.agent_pos[i]
        rel_angle = self._wrap(math.atan2(dy, dx) - self.agent_theta[i]) / math.pi
        
        dist_food = np.linalg.norm(self.agent_pos[i] - self.food_pos)
        odor = float(np.exp(-dist_food / 40.0)) if dist_food <= self.config.odor_radius else 0.0
        
        # Soft noise in scent 
        if dist_food < 40.0:
            odor = float(np.clip(odor + np.random.normal(0, 0.01), 0.0, 1.0))
            
        # 3. Construction of the baseline observation
        observation = prox + [rel_angle, odor]
        
        # Social Channel (Ablations and Shuffling)
        if self.config.num_agents > 1:
            other_idx = 1 - i
            social_signal = 0.0

            if self.config.social_mode != "off" and self.agent_signals[other_idx] > 0.5:
                dx_s, dy_s = self.agent_pos[other_idx] - self.agent_pos[i]
                dist_s = np.linalg.norm([dx_s, dy_s])
                rel_angle_social = self._wrap(math.atan2(dy_s, dx_s) - self.agent_theta[i]) / math.pi
                signal_strength = float(np.exp(-dist_s / 180.0))

                # Baseline Variant
                if self.config.social_variant == "angle_only":
                    base_signal = float(rel_angle_social)
                elif self.config.social_variant == "strength_only":
                    base_signal = float(signal_strength)
                else: # angle_strength
                    base_signal = float(rel_angle_social * signal_strength)

                # Shuffling/Contaminate
                if self.config.social_mode == "shuffled":
                    if self.config.social_variant == "angle_only":
                        social_signal = float(np.random.uniform(-1.0, 1.0))
                    elif self.config.social_variant == "strength_only":
                        social_signal = float(np.random.uniform(0.0, 1.0))
                    else:
                        fake_angle = float(np.random.uniform(-1.0, 1.0))
                        social_signal = float(fake_angle * signal_strength)
                else:
                    social_signal = base_signal

            observation.append(float(social_signal))
                
        return np.array(observation, dtype=np.float32)

    def _get_obs_all(self): return [self._get_obs_single(i) for i in range(self.config.num_agents)]
    
    def _raycast(self, agent_idx, angle) -> float:
        obs_array = np.array(self.obstacles, dtype=np.float32)
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        dist = fast_raycast(self.agent_pos[agent_idx], direction, obs_array, 
                           float(self.config.world_width), float(self.config.world_height), 
                           float(self.config.sensor_range))
        
        if self.config.num_agents > 1:
            other_idx = 1 - agent_idx
            # We use the same 0.9 margin so that the view matches the physics.
            dist_to_other = self._ray_circle_intersect(self.agent_pos[agent_idx], direction, 
                                                     self.agent_pos[other_idx], self.config.collision_radius * 0.9)
            dist = min(dist, dist_to_other)
        return dist

    def _ray_circle_intersect(self, start, dir, center, radius):
        L = center - start
        tc = np.dot(L, dir)
        if tc < 0: return self.config.sensor_range
        d2 = np.dot(L, L) - tc*tc
        if d2 > radius*radius: return self.config.sensor_range
        t1c = math.sqrt(radius*radius - d2)
        return float(tc - t1c) if (tc - t1c) < self.config.sensor_range else self.config.sensor_range

    def _is_success(self, i) -> bool: return np.linalg.norm(self.agent_pos[i] - self.food_pos) < self.config.success_threshold

    def _collides(self, i, p):
        if p[0] < 20 or p[0] > 780 or p[1] < 20 or p[1] > 580: return True
        for ox, oy, ow, oh in self.obstacles:
            nx, ny = np.clip(p[0], ox, ox+ow), np.clip(p[1], oy, oy+oh)
            # USE collision_radius
            if np.linalg.norm(p - [nx, ny]) <= self.config.collision_radius: return True
        
        # --- Collision between agents with noise and margin ---
        if i >= 0 and self.config.num_agents > 1:
            other_idx = 1 - i
            #  We inject 5px of noise into the perceived position of the partner to break the block
            jitter_pos = self.agent_pos[other_idx] + np.random.normal(0, 5.0, size=2)
            dist_to_other = np.linalg.norm(p - jitter_pos)
            
            # We lowered the multiplier from 1.1 to 0.9 (allows for "grazing" contact without a full block)
            if dist_to_other <= self.config.collision_radius * 0.9: 
                return True
        return False

    def _wrap(self, a): return float((a + math.pi) % (2.0 * math.pi) - math.pi)
    def _get_info(self):
        success_steps = list(self.agent_success_step)
        valid_steps = [s for s in success_steps if s >= 0]

        first_goal_step = min(valid_steps) if len(valid_steps) > 0 else -1
        second_goal_step = max(valid_steps) if len(valid_steps) == self.config.num_agents else -1
        goal_gap = (second_goal_step - first_goal_step) if len(valid_steps) == self.config.num_agents else -1

        return {
            "success": all(self.agent_success),
            "individual_success": self.agent_success,
            "step_count": self.step_count,
            "success_steps": success_steps,
            "first_goal_step": first_goal_step,
            "second_goal_step": second_goal_step,
            "goal_gap": goal_gap,
            "signal_counts": self.signal_counts.tolist(),
        }

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.config.world_width, self.config.world_height))
            self.clock = pygame.time.Clock()
        self.window.fill((235, 235, 235))
        overlay = pygame.Surface((self.config.world_width, self.config.world_height), pygame.SRCALPHA)
        pygame.draw.circle(overlay, (50, 200, 50, 30), self.food_pos.astype(int), int(self.config.odor_radius))
        pygame.draw.circle(overlay, (50, 200, 50, 60), self.food_pos.astype(int), int(self.config.success_threshold))
        for o in self.obstacles: pygame.draw.rect(self.window, (80, 80, 80), pygame.Rect(*o))
        pygame.draw.circle(self.window, (30, 160, 30), self.food_pos.astype(int), int(self.config.food_radius))
               
        # --- DRAW TRAJECTORY TRAILS ---
        for i in range(self.config.num_agents):
            if len(self.agent_paths[i]) >= 2: # Pygame necesita al menos 2 puntos
                # Color del rastro (más suave que el robot)
                path_color = (150, 180, 240) if i == 0 else (210, 150, 240)
                # Forzamos la conversión a lista de pares de números
                points = [(p[0], p[1]) for p in self.agent_paths[i]]
                pygame.draw.lines(self.window, path_color, False, points, 2)
        
        for i in range(self.config.num_agents): # Live Path Tracing
            color = (40, 110, 210) if i == 0 else (160, 40, 190)
            pos_i = self.agent_pos[i].astype(int)
            
            # --- RAYTRACING ---
            angle_step = 2.0 * math.pi / self.config.num_sensors
            for j in range(self.config.num_sensors):
                angle = self.agent_theta[i] + j * angle_step
                dist = self._raycast(i, angle)
                end_p = self.agent_pos[i] + np.array([math.cos(angle), math.sin(angle)]) * dist
                pygame.draw.line(self.window, (200, 200, 200), pos_i, end_p.astype(int), 1)

            if self.agent_signals[i] > 0.5 and (self.step_count % 2 == 0):
                pygame.draw.circle(overlay, (255, 150, 0, 110), pos_i, 42)
            
            pygame.draw.circle(self.window, color, pos_i, int(self.config.robot_radius))
            end_dir = self.agent_pos[i] + np.array([np.cos(self.agent_theta[i]), np.sin(self.agent_theta[i])]) * 15
            pygame.draw.line(self.window, (30, 30, 30), pos_i, end_dir.astype(int), 3)
        self.window.blit(overlay, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.config.render_fps)
        return pygame.surfarray.array3d(self.window).transpose(1, 0, 2)
        
    def close(self):
        if self.window: pygame.quit(); self.window = None
