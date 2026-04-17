import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrow
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from stable_baselines3 import SAC
from f1_2026_env import F12026Env
import os
import sys

# 1. Load environment and model
env = F12026Env(track_length=5000)

model_path = "sac_f1_2026_2d.zip"
if not os.path.exists(model_path):
    best_path = "best_model/best_model.zip"
    if os.path.exists(best_path):
        model_path = best_path
    else:
        print(f"No model found at {model_path} or {best_path}")
        print("Run train_f1.py first, or provide a model path as argument.")
        sys.exit(1)

print(f"Loading model from {model_path}")
model = SAC.load(model_path, env=env)

# 2. Run evaluation episode
obs, _ = env.reset()
history = []
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    x, y, psi = env.get_car_pose()
    history.append({
        'x': x, 'y': y, 'psi': psi,
        'soc': info['soc'],
        'vel': info['velocity_kmh'],
        'grip': info['grip_ratio'],
        'lateral': info['lateral_offset'],
        'progress': info['progress_s'],
    })
    done = terminated or truncated

print(f"Episode: {len(history)} steps, time={info['time']:.1f}s, "
      f"lap_complete={info.get('lap_complete', False)}")

# 3. Setup figure
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Draw track surface (filled between boundaries)
lx, ly, rx, ry = env.get_track_boundaries()
# Draw as filled polygon: go along left boundary, then back along right
track_poly_x = np.concatenate([lx, rx[::-1], [lx[0]]])
track_poly_y = np.concatenate([ly, ry[::-1], [ly[0]]])
ax.fill(track_poly_x, track_poly_y, color='#2a2a3e', zorder=1)

# Track edges
ax.plot(lx, ly, color='#ffffff', linewidth=1.0, alpha=0.6, zorder=2)
ax.plot(rx, ry, color='#ffffff', linewidth=1.0, alpha=0.6, zorder=2)

# Centerline (dashed)
cx, cy = env.get_centerline()
ax.plot(cx, cy, color='#555580', linewidth=0.5, linestyle='--', alpha=0.4, zorder=2)

# Start/finish marker
ax.plot(cx[0], cy[0], 's', color='white', markersize=8, zorder=10)
ax.annotate('S/F', (cx[0], cy[0]), color='white', fontsize=9, fontweight='bold',
            xytext=(10, 10), textcoords='offset points', zorder=10)

ax.set_aspect('equal')
ax.set_title("F1 2026 Energy Management — 2D Lap Trace", color='white',
             fontsize=14, fontweight='bold')
ax.tick_params(colors='#666666')
for spine in ax.spines.values():
    spine.set_color('#333333')

# Speed colormap
vel_array = np.array([h['vel'] for h in history])
vel_min, vel_max = vel_array.min(), max(vel_array.max(), vel_array.min() + 1)
cmap = plt.cm.plasma
norm = mcolors.Normalize(vmin=vel_min, vmax=vel_max)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label('Speed (km/h)', color='white', fontsize=10)
cbar.ax.tick_params(colors='white')

# Dynamic elements
car_arrow = None
trail_collection = None

text_style = dict(transform=ax.transAxes, fontsize=10, fontfamily='monospace',
                  color='white', verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                            edgecolor='#555580', alpha=0.9))
soc_text = ax.text(0.02, 0.98, '', **text_style)
vel_text = ax.text(0.02, 0.92, '', **text_style)
step_text = ax.text(0.02, 0.86, '', **text_style)
grip_text = ax.text(0.02, 0.80, '', **text_style)


def init():
    soc_text.set_text('')
    vel_text.set_text('')
    step_text.set_text('')
    grip_text.set_text('')
    return soc_text, vel_text, step_text, grip_text


def update(frame):
    global car_arrow, trail_collection
    h = history[frame]

    # Remove old car arrow
    if car_arrow is not None:
        car_arrow.remove()

    # Draw car as arrow showing heading
    arrow_len = 25.0
    dx = arrow_len * np.cos(h['psi'])
    dy = arrow_len * np.sin(h['psi'])
    car_arrow = ax.annotate('', xy=(h['x'] + dx, h['y'] + dy),
                            xytext=(h['x'], h['y']),
                            arrowprops=dict(arrowstyle='->', color='#ff4444',
                                            lw=2.5, mutation_scale=15),
                            zorder=20)

    # Car dot
    # (drawn as part of trail head)

    # Text overlays
    soc_pct = h['soc'] * 100
    soc_bars = int(soc_pct / 5)
    soc_bar_str = '█' * soc_bars + '░' * (20 - soc_bars)
    soc_text.set_text(f"SoC: {soc_pct:5.1f}% [{soc_bar_str}]")
    vel_text.set_text(f"Speed: {h['vel']:5.1f} km/h")
    step_text.set_text(f"Step: {frame}/{len(history)}")
    grip_text.set_text(f"Grip: {h['grip']:.2f}" +
                       (" ⚠" if h['grip'] > 1.0 else ""))

    # Speed-colored trail
    if trail_collection is not None:
        trail_collection.remove()
        trail_collection = None

    trail_len = 80
    start = max(0, frame - trail_len)
    if frame > start + 1:
        trail_x = [history[i]['x'] for i in range(start, frame + 1)]
        trail_y = [history[i]['y'] for i in range(start, frame + 1)]
        trail_v = [history[i]['vel'] for i in range(start, frame + 1)]

        points = np.array([trail_x, trail_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5, zorder=5)
        lc.set_array(np.array(trail_v[:-1]))
        trail_collection = ax.add_collection(lc)

    return soc_text, vel_text, step_text, grip_text


# Downsample for manageable GIF
step_size = max(1, len(history) // 400)  # target ~400 frames max
frames = list(range(0, len(history), step_size))

ani = FuncAnimation(fig, update, frames=frames,
                    init_func=init, blit=False, interval=50)

print(f"\nGenerating GIF ({len(frames)} frames)...")
ani.save('f1_2026_lap.gif', writer='pillow', fps=20)
print("Animation saved as 'f1_2026_lap.gif'")
plt.close()

# 4. Static racing line comparison plot
fig2, ax2 = plt.subplots(figsize=(14, 10))
fig2.patch.set_facecolor('#1a1a2e')
ax2.set_facecolor('#1a1a2e')

# Track surface
ax2.fill(track_poly_x, track_poly_y, color='#2a2a3e', zorder=1)
ax2.plot(lx, ly, color='#ffffff', linewidth=1.0, alpha=0.6, zorder=2)
ax2.plot(rx, ry, color='#ffffff', linewidth=1.0, alpha=0.6, zorder=2)
ax2.plot(cx, cy, color='#aaaacc', linewidth=1.5, linestyle='--', alpha=0.6,
         zorder=3, label='Centerline')

# Agent racing line colored by speed
all_x = [h['x'] for h in history]
all_y = [h['y'] for h in history]
all_v = [h['vel'] for h in history]

points = np.array([all_x, all_y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5, zorder=5)
lc.set_array(np.array(all_v[:-1]))
ax2.add_collection(lc)

# Colorbar
sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm2.set_array([])
cbar2 = fig2.colorbar(sm2, ax=ax2, shrink=0.5, pad=0.02)
cbar2.set_label('Speed (km/h)', color='white', fontsize=10)
cbar2.ax.tick_params(colors='white')

ax2.set_aspect('equal')
ax2.set_title("F1 2026 — Learned Racing Line vs Centerline", color='white',
              fontsize=14, fontweight='bold')
ax2.tick_params(colors='#666666')
for spine in ax2.spines.values():
    spine.set_color('#333333')

centerline_patch = mpatches.Patch(color='#aaaacc', label='Centerline')
agent_patch = mpatches.Patch(color='#ff6600', label='Agent line')
ax2.legend(handles=[centerline_patch, agent_patch], loc='upper right',
           facecolor='#1a1a2e', edgecolor='#555580', labelcolor='white')

plt.tight_layout()
plt.savefig('f1_2026_racing_line.png', dpi=150)
print("Racing line comparison saved as 'f1_2026_racing_line.png'")
plt.close()
