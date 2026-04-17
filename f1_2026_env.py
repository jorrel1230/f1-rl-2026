import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class F12026Env(gym.Env):
    """
    Full 2D F1 2026 Energy Management Environment.
    Bicycle model with steering, friction circle, and 2026-regulation energy system.
    Agent controls: steering, throttle, brake, MGU-K deploy/harvest.
    """

    def __init__(self, track_length=5000, track_width=12.0):
        super().__init__()

        self.track_length = track_length
        self.dt = 0.05  # 50ms time step (smaller for steering stability)
        self.max_steps = 20_000  # 1000s max episode

        # --- Action space: [steering, throttle, brake, mguk_deploy] ---
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # --- Observation space: 12D normalized ---
        self.observation_space = spaces.Box(
            low=-np.ones(12, dtype=np.float32),
            high=np.ones(12, dtype=np.float32),
            dtype=np.float32,
        )

        # --- Car constants (2026 regs) ---
        self.MGU_K_MAX_POWER = 350_000.0    # 350 kW
        self.BATTERY_CAPACITY_J = 4_000_000.0  # 4 MJ
        self.MAX_HARVEST_PER_LAP_J = 8_500_000.0  # 8.5 MJ per lap
        self.ICE_POWER = 400_000.0           # 400 kW
        self.CAR_MASS = 768.0                # kg (2026 minimum)
        self.AIR_DENSITY = 1.225
        self.CDA = 1.2
        self.WHEELBASE = 3.6                 # meters
        self.MAX_STEER = 0.35                # rad (~20 degrees)
        self.MU = 2.5                        # tire friction coefficient
        self.G = 9.81
        self.BRAKE_FORCE_MAX = 20_000.0      # N
        self.REGEN_EFFICIENCY = 0.8
        self.TRACK_WIDTH = track_width
        self.V_MAX = 100.0                   # ~360 km/h normalization reference
        self.MAX_YAW_RATE = 2.0              # rad/s normalization reference
        self.KAPPA_MAX = 0.05                # curvature normalization (20m radius)

        # Curvature lookahead distances (meters ahead)
        self.LOOKAHEAD_DISTS = np.array([20.0, 50.0, 100.0, 200.0, 400.0])

        # Build track
        self._init_track_2d()

    # ------------------------------------------------------------------ #
    #  Track infrastructure                                                #
    # ------------------------------------------------------------------ #
    def _init_track_2d(self):
        """Build closed circuit with spline, boundaries, and spatial index."""
        waypoints = np.array([
            [0, 0],
            [600, 0],       # long main straight
            [700, 50],      # turn 1 entry
            [720, 150],     # turn 1 (right-hander)
            [650, 220],     # turn 2
            [500, 250],     # short straight
            [350, 300],     # turn 3 (chicane entry)
            [300, 250],     # chicane mid
            [250, 300],     # chicane exit
            [100, 350],     # back straight approach
            [-100, 400],    # turn 5
            [-200, 350],    # turn 6
            [-250, 200],    # back straight
            [-300, 50],     # turn 7
            [-280, -50],    # turn 8
            [-200, -100],   # turn 9
            [-50, -80],     # approach to final
            [0, 0],         # close the loop
        ])

        # Parameterize by cumulative chord length
        diffs = np.diff(waypoints, axis=0)
        chord_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        t_wp = np.concatenate(([0], np.cumsum(chord_lengths)))

        # Periodic cubic spline
        n_pts = 2000
        cs_x = CubicSpline(t_wp, waypoints[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t_wp, waypoints[:, 1], bc_type='periodic')

        t_fine = np.linspace(0, t_wp[-1], n_pts, endpoint=False)
        self.track_x = cs_x(t_fine)
        self.track_y = cs_y(t_fine)
        self._n_track_pts = n_pts

        # Cumulative arc length
        dx = np.diff(self.track_x)
        dy = np.diff(self.track_y)
        seg_len = np.sqrt(dx ** 2 + dy ** 2)
        self.track_cum_dist = np.concatenate(([0], np.cumsum(seg_len)))
        # Scale to track_length
        self.track_cum_dist *= self.track_length / self.track_cum_dist[-1]

        # Unit tangent vectors
        tx = cs_x(t_fine, 1)
        ty = cs_y(t_fine, 1)
        t_mag = np.sqrt(tx ** 2 + ty ** 2)
        self.tangent_x = tx / t_mag
        self.tangent_y = ty / t_mag

        # Track heading at each point
        self.track_heading = np.arctan2(self.tangent_y, self.tangent_x)

        # Unit normals (90° CCW from tangent → left side of track)
        self.normal_x = -self.tangent_y
        self.normal_y = self.tangent_x

        # Track boundaries
        hw = self.TRACK_WIDTH / 2.0
        self.left_x = self.track_x + hw * self.normal_x
        self.left_y = self.track_y + hw * self.normal_y
        self.right_x = self.track_x - hw * self.normal_x
        self.right_y = self.track_y - hw * self.normal_y

        # Signed curvature
        ddx = cs_x(t_fine, 2)
        ddy = cs_y(t_fine, 2)
        self.signed_curvature = (tx * ddy - ty * ddx) / (t_mag ** 3)

        # KD-tree for fast closest-point lookup
        self._centerline_pts = np.column_stack([self.track_x, self.track_y])
        self._kdtree = cKDTree(self._centerline_pts)

    def _find_closest(self, x, y):
        """Find closest centerline point. Returns (arc_length_s, signed_lateral_offset_d, track_heading)."""
        _, idx = self._kdtree.query([x, y])

        # Refine: project onto segment between neighbors
        idx_prev = (idx - 1) % self._n_track_pts
        idx_next = (idx + 1) % self._n_track_pts

        best_s = self.track_cum_dist[idx]
        best_d = self._signed_offset(x, y, idx)

        # Check previous segment
        for nb in [idx_prev, idx_next]:
            s_nb, d_nb = self._project_on_segment(x, y, nb, idx)
            if abs(d_nb) < abs(best_d):
                best_s = s_nb
                best_d = d_nb

        heading = self.track_heading[idx]
        return best_s, best_d, heading

    def _signed_offset(self, x, y, idx):
        """Signed lateral offset: positive = left of centerline."""
        dx = x - self.track_x[idx]
        dy = y - self.track_y[idx]
        return dx * self.normal_x[idx] + dy * self.normal_y[idx]

    def _project_on_segment(self, x, y, idx_a, idx_b):
        """Project point onto segment a→b, return (arc_length, signed_offset)."""
        ax, ay = self.track_x[idx_a], self.track_y[idx_a]
        bx, by = self.track_x[idx_b], self.track_y[idx_b]
        abx, aby = bx - ax, by - ay
        apx, apy = x - ax, y - ay
        ab_len2 = abx ** 2 + aby ** 2
        if ab_len2 < 1e-10:
            return self.track_cum_dist[idx_a], self._signed_offset(x, y, idx_a)
        t = np.clip((apx * abx + apy * aby) / ab_len2, 0.0, 1.0)
        s_a = self.track_cum_dist[idx_a]
        s_b = self.track_cum_dist[idx_b]
        # Handle wrap
        ds = s_b - s_a
        if abs(ds) > self.track_length / 2:
            ds = ds - np.sign(ds) * self.track_length
        s_proj = (s_a + t * ds) % self.track_length
        # Offset at projected point (use nearest discrete index normal)
        mid_idx = idx_a if t < 0.5 else idx_b
        d_proj = self._signed_offset(x, y, mid_idx)
        return s_proj, d_proj

    def _get_curvature_lookahead(self, s):
        """Return signed curvature at lookahead distances ahead of s."""
        result = np.zeros(len(self.LOOKAHEAD_DISTS))
        for i, dist in enumerate(self.LOOKAHEAD_DISTS):
            s_ahead = (s + dist) % self.track_length
            idx = np.searchsorted(self.track_cum_dist, s_ahead, side='right') - 1
            idx = np.clip(idx, 0, self._n_track_pts - 1)
            result[i] = self.signed_curvature[idx]
        return result

    # ------------------------------------------------------------------ #
    #  Observation                                                         #
    # ------------------------------------------------------------------ #
    def _build_obs(self):
        """Build normalized 12D observation."""
        x, y, psi, vel, soc = self.state
        s, d, track_h = self._find_closest(x, y)
        self._last_s = s
        self._last_d = d

        heading_error = wrap_angle(psi - track_h)
        hw = self.TRACK_WIDTH / 2.0
        curvature_ahead = self._get_curvature_lookahead(s)

        obs = np.array([
            vel / self.V_MAX,                                    # [0] velocity
            soc,                                                 # [1] SoC
            np.clip(d / hw, -1.0, 1.0),                         # [2] lateral offset
            heading_error / np.pi,                               # [3] heading error
            *np.clip(curvature_ahead / self.KAPPA_MAX, -1, 1),  # [4-8] curvature lookahead
            (self.MAX_HARVEST_PER_LAP_J - self.total_harvested_j) / self.MAX_HARVEST_PER_LAP_J,  # [9]
            np.clip(self._yaw_rate / self.MAX_YAW_RATE, -1, 1), # [10] yaw rate
            s / self.track_length,                               # [11] progress
        ], dtype=np.float32)
        return obs

    # ------------------------------------------------------------------ #
    #  Reset / Step                                                        #
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start at track origin, aligned with track heading
        x0 = self.track_x[0]
        y0 = self.track_y[0]
        psi0 = self.track_heading[0]
        vel0 = 30.0  # rolling start m/s (~108 km/h)
        soc0 = 1.0

        self.state = np.array([x0, y0, psi0, vel0, soc0], dtype=np.float64)
        self.time_elapsed = 0.0
        self.total_harvested_j = 0.0
        self._yaw_rate = 0.0
        self._step_count = 0
        self._last_s = 0.0
        self._last_d = 0.0
        self._prev_s = 0.0
        self._milestones_hit = set()

        return self._build_obs(), {}

    def step(self, action):
        steering_raw, throttle, brake, mguk_request = action
        x, y, psi, vel, soc = self.state

        delta = float(steering_raw) * self.MAX_STEER  # steering angle
        throttle = float(np.clip(throttle, 0, 1))
        brake = float(np.clip(brake, 0, 1))
        mguk_request = float(np.clip(mguk_request, -1, 1))

        # ---- 1. Energy management (2026 regs) ----
        harvest_headroom = self.MAX_HARVEST_PER_LAP_J - self.total_harvested_j

        if mguk_request > 0:  # Deploying
            actual_mguk_power = min(
                mguk_request * self.MGU_K_MAX_POWER,
                (soc * self.BATTERY_CAPACITY_J) / self.dt,
            )
        else:  # Harvesting — respect per-lap cap
            desired_harvest = -mguk_request * self.MGU_K_MAX_POWER * self.dt
            allowed_harvest = min(desired_harvest, max(harvest_headroom, 0.0))
            actual_mguk_power = -allowed_harvest / self.dt

        # SoC update from MGU-K
        energy_change = -actual_mguk_power * self.dt
        if energy_change > 0:
            self.total_harvested_j += energy_change
        new_soc = np.clip(soc + energy_change / self.BATTERY_CAPACITY_J, 0.0, 1.0)

        # ---- 2. Longitudinal forces ----
        total_power = throttle * self.ICE_POWER + actual_mguk_power
        tractive_force = total_power / max(vel, 1.0)
        braking_force = brake * self.BRAKE_FORCE_MAX

        drag_force = 0.5 * self.AIR_DENSITY * self.CDA * vel ** 2
        rolling_resistance = 500.0

        Fx_net = tractive_force - braking_force - drag_force - rolling_resistance
        ax_long = Fx_net / self.CAR_MASS

        # ---- 3. Bicycle model kinematics ----
        beta = np.arctan(0.5 * np.tan(delta))  # slip angle (lr/L ≈ 0.5)
        yaw_rate = (vel / self.WHEELBASE) * np.sin(beta) * 2.0
        self._yaw_rate = yaw_rate

        # ---- 4. Friction circle ----
        Fy_lat = self.CAR_MASS * vel ** 2 * np.tan(delta) / self.WHEELBASE
        F_total = np.sqrt(Fx_net ** 2 + Fy_lat ** 2)
        F_max = self.MU * self.CAR_MASS * self.G
        grip_ratio = F_total / max(F_max, 1.0)
        grip_exceeded = grip_ratio > 1.0

        if grip_exceeded:
            scale = F_max / F_total
            ax_long *= scale
            yaw_rate *= scale
            vel_penalty = (grip_ratio - 1.0) * self.MU * self.G * self.dt
        else:
            vel_penalty = 0.0

        new_vel = max(vel + ax_long * self.dt - vel_penalty, 1.0)

        # ---- 5. Position update ----
        new_psi = wrap_angle(psi + yaw_rate * self.dt)
        new_x = x + vel * np.cos(psi + beta) * self.dt
        new_y = y + vel * np.sin(psi + beta) * self.dt

        # ---- 6. Regen: brake + deceleration + coasting ----
        # Real F1: MGU-K harvests from any deceleration event at rear axle
        # Sources: braking, lift-off (partial throttle), coasting, grip loss
        decel = vel - new_vel  # positive when slowing down
        regen_power = 0.0

        if decel > 0:
            # Kinetic energy lost this step, recover portion via MGU-K
            ke_lost = 0.5 * self.CAR_MASS * (vel ** 2 - new_vel ** 2)
            regen_power = (ke_lost / self.dt) * self.REGEN_EFFICIENCY

        # Cap at MGU-K max harvest rate (350 kW)
        regen_power = min(regen_power, self.MGU_K_MAX_POWER)
        regen_energy = regen_power * self.dt

        # Respect per-lap harvest cap
        harvest_headroom = self.MAX_HARVEST_PER_LAP_J - self.total_harvested_j
        regen_energy = min(regen_energy, max(harvest_headroom, 0.0))
        self.total_harvested_j += regen_energy
        new_soc = np.clip(new_soc + regen_energy / self.BATTERY_CAPACITY_J, 0.0, 1.0)

        # ---- 7. Update state ----
        self.state = np.array([new_x, new_y, new_psi, new_vel, new_soc], dtype=np.float64)
        self.time_elapsed += self.dt
        self._step_count += 1

        # ---- 8. Track position & reward ----
        obs = self._build_obs()
        new_s = self._last_s
        new_d = self._last_d
        hw = self.TRACK_WIDTH / 2.0

        # Progress (handle wraparound)
        delta_s = new_s - self._prev_s
        if delta_s < -self.track_length / 2:
            delta_s += self.track_length
        elif delta_s > self.track_length / 2:
            delta_s -= self.track_length

        # Reward components
        progress_reward = (delta_s / self.track_length) * 100.0
        lateral_penalty = -0.5 * (new_d / hw) ** 2
        grip_penalty = -5.0 * max(grip_ratio - 1.0, 0.0)
        alive_bonus = 0.1

        reward = progress_reward + lateral_penalty + grip_penalty + alive_bonus

        # Progress milestones — encourage getting further around track
        progress_pct = new_s / self.track_length
        for milestone in [0.25, 0.50, 0.75, 0.90]:
            if milestone not in self._milestones_hit and progress_pct >= milestone:
                reward += 20.0
                self._milestones_hit.add(milestone)

        # Off-track check
        off_track = abs(new_d) > hw
        terminated = False
        truncated = False

        if off_track:
            reward = -50.0
            terminated = True

        # Lap complete
        lap_complete = (self._prev_s > self.track_length * 0.9 and new_s < self.track_length * 0.1 and delta_s > 0)
        if not lap_complete and delta_s > 0:
            pass
        if lap_complete:
            reward += 1000.0 / self.time_elapsed  # bumped from 500
            terminated = True

        # Truncation
        if self._step_count >= self.max_steps:
            truncated = True

        # Going backwards too much
        if delta_s < -10.0:
            reward -= 10.0

        self._prev_s = new_s

        info = {
            "time": self.time_elapsed,
            "grip_ratio": grip_ratio,
            "lateral_offset": new_d,
            "progress_s": new_s,
            "lap_complete": lap_complete,
            "soc": new_soc,
            "velocity_kmh": new_vel * 3.6,
        }

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Visualization helpers                                               #
    # ------------------------------------------------------------------ #
    def get_car_pose(self):
        """Return (x, y, heading) for visualization."""
        return self.state[0], self.state[1], self.state[2]

    def get_track_boundaries(self):
        """Return (left_x, left_y, right_x, right_y)."""
        return self.left_x, self.left_y, self.right_x, self.right_y

    def get_centerline(self):
        """Return (track_x, track_y)."""
        return self.track_x, self.track_y
