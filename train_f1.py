import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

from f1_2026_env import F12026Env


def find_latest_checkpoint(checkpoint_dir="./checkpoints", prefix="sac_f1"):
    """Find most recent checkpoint and return (path, step_count)."""
    if not os.path.exists(checkpoint_dir):
        return None, 0
    pattern = re.compile(rf"^{prefix}_(\d+)_steps\.zip$")
    best_step = 0
    best_path = None
    for f in os.listdir(checkpoint_dir):
        m = pattern.match(f)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_path = os.path.join(checkpoint_dir, f)
    return best_path, best_step


# ------------------------------------------------------------------ #
#  Custom callbacks                                                    #
# ------------------------------------------------------------------ #

class TqdmCallback(BaseCallback):
    """tqdm progress bar for SB3 training."""

    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="step")

    def _on_step(self):
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()


class EvalGifCallback(BaseCallback):
    """Run eval episode and save GIF + racing line PNG at fixed intervals."""

    def __init__(self, eval_env, gif_dir="eval_gifs", gif_freq=50_000):
        super().__init__()
        self.eval_env = eval_env
        self.gif_dir = gif_dir
        self.gif_freq = gif_freq
        os.makedirs(gif_dir, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.gif_freq == 0:
            step_k = self.num_timesteps // 1000
            print(f"\n[EvalGif] Generating eval at {step_k}k steps...")
            self._generate_eval(step_k)
        return True

    def _generate_eval(self, step_k):
        from matplotlib.animation import FuncAnimation

        env = self.eval_env.unwrapped if hasattr(self.eval_env, 'unwrapped') else self.eval_env
        obs, _ = self.eval_env.reset()
        history = []
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            x, y, psi = env.get_car_pose()
            history.append({
                'x': x, 'y': y, 'psi': psi,
                'soc': info['soc'],
                'vel': info['velocity_kmh'],
                'grip': info['grip_ratio'],
            })
            done = terminated or truncated

        if len(history) < 5:
            print(f"  [EvalGif] Episode too short ({len(history)} steps), skipping.")
            return

        lap_status = "complete" if info.get('lap_complete', False) else "incomplete"
        print(f"  [EvalGif] {len(history)} steps, {info['time']:.1f}s, {lap_status}")

        # --- Setup figure ---
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        lx, ly, rx, ry = env.get_track_boundaries()
        cx, cy = env.get_centerline()

        # Track surface
        poly_x = np.concatenate([lx, rx[::-1], [lx[0]]])
        poly_y = np.concatenate([ly, ry[::-1], [ly[0]]])
        ax.fill(poly_x, poly_y, color='#2a2a3e', zorder=1)
        ax.plot(lx, ly, color='#ffffff', linewidth=1.0, alpha=0.6, zorder=2)
        ax.plot(rx, ry, color='#ffffff', linewidth=1.0, alpha=0.6, zorder=2)
        ax.plot(cx, cy, color='#555580', linewidth=0.5, linestyle='--', alpha=0.4, zorder=2)
        ax.plot(cx[0], cy[0], 's', color='white', markersize=8, zorder=10)

        ax.set_aspect('equal')
        ax.set_title(f"F1 2026 — Eval at {step_k}k steps ({lap_status})",
                     color='white', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#666666')
        for sp in ax.spines.values():
            sp.set_color('#333333')

        vel_arr = np.array([h['vel'] for h in history])
        cmap = plt.cm.plasma
        norm = mcolors.Normalize(vmin=vel_arr.min(), vmax=max(vel_arr.max(), vel_arr.min() + 1))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label('Speed (km/h)', color='white', fontsize=10)
        cbar.ax.tick_params(colors='white')

        car_arrow = [None]
        trail_coll = [None]

        txt_style = dict(transform=ax.transAxes, fontsize=10, fontfamily='monospace',
                         color='white', verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                                   edgecolor='#555580', alpha=0.9))
        soc_txt = ax.text(0.02, 0.98, '', **txt_style)
        vel_txt = ax.text(0.02, 0.92, '', **txt_style)
        step_txt = ax.text(0.02, 0.86, '', **txt_style)

        def update(frame):
            h = history[frame]
            if car_arrow[0] is not None:
                car_arrow[0].remove()
            alen = 25.0
            car_arrow[0] = ax.annotate(
                '', xy=(h['x'] + alen * np.cos(h['psi']), h['y'] + alen * np.sin(h['psi'])),
                xytext=(h['x'], h['y']),
                arrowprops=dict(arrowstyle='->', color='#ff4444', lw=2.5, mutation_scale=15),
                zorder=20)

            soc_pct = h['soc'] * 100
            bars = int(soc_pct / 5)
            soc_txt.set_text(f"SoC: {soc_pct:5.1f}% [{'█' * bars}{'░' * (20 - bars)}]")
            vel_txt.set_text(f"Speed: {h['vel']:5.1f} km/h")
            step_txt.set_text(f"Step: {frame}/{len(history)}")

            if trail_coll[0] is not None:
                trail_coll[0].remove()
                trail_coll[0] = None
            tl = 80
            s = max(0, frame - tl)
            if frame > s + 1:
                tx = [history[i]['x'] for i in range(s, frame + 1)]
                ty = [history[i]['y'] for i in range(s, frame + 1)]
                tv = [history[i]['vel'] for i in range(s, frame + 1)]
                pts = np.array([tx, ty]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=2.5, zorder=5)
                lc.set_array(np.array(tv[:-1]))
                trail_coll[0] = ax.add_collection(lc)

            return soc_txt, vel_txt, step_txt

        step_size = max(1, len(history) // 300)
        frames = list(range(0, len(history), step_size))
        ani = FuncAnimation(fig, update, frames=frames, blit=False, interval=50)

        gif_path = os.path.join(self.gif_dir, f"eval_{step_k:04d}k.gif")
        ani.save(gif_path, writer='pillow', fps=20)
        plt.close(fig)
        print(f"  [EvalGif] Saved {gif_path}")

        # --- Static racing line PNG ---
        fig2, ax2 = plt.subplots(figsize=(12, 9))
        fig2.patch.set_facecolor('#1a1a2e')
        ax2.set_facecolor('#1a1a2e')
        ax2.fill(poly_x, poly_y, color='#2a2a3e', zorder=1)
        ax2.plot(lx, ly, color='#ffffff', linewidth=1.0, alpha=0.6, zorder=2)
        ax2.plot(rx, ry, color='#ffffff', linewidth=1.0, alpha=0.6, zorder=2)
        ax2.plot(cx, cy, color='#aaaacc', linewidth=1.5, linestyle='--', alpha=0.6, zorder=3)

        all_x = [h['x'] for h in history]
        all_y = [h['y'] for h in history]
        all_v = [h['vel'] for h in history]
        pts = np.array([all_x, all_y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=2.5, zorder=5)
        lc.set_array(np.array(all_v[:-1]))
        ax2.add_collection(lc)

        sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm2.set_array([])
        cbar2 = fig2.colorbar(sm2, ax=ax2, shrink=0.5, pad=0.02)
        cbar2.set_label('Speed (km/h)', color='white', fontsize=10)
        cbar2.ax.tick_params(colors='white')

        ax2.set_aspect('equal')
        ax2.set_title(f"Racing Line — {step_k}k steps ({lap_status})",
                      color='white', fontsize=13, fontweight='bold')
        ax2.tick_params(colors='#666666')
        for sp in ax2.spines.values():
            sp.set_color('#333333')

        png_path = os.path.join(self.gif_dir, f"line_{step_k:04d}k.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        plt.close(fig2)
        print(f"  [EvalGif] Saved {png_path}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

parser = argparse.ArgumentParser(description="Train F1 2026 SAC agent")
parser.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint")
parser.add_argument("--total-steps", type=int, default=500_000,
                    help="Total training timesteps (default: 500k)")
parser.add_argument("--gif-freq", type=int, default=50_000,
                    help="Eval GIF frequency in steps (default: 50k)")
args = parser.parse_args()

TOTAL_TIMESTEPS = args.total_steps
GIF_FREQ = args.gif_freq

# 1. Environments
env = Monitor(F12026Env(track_length=5000))
eval_env = Monitor(F12026Env(track_length=5000))
gif_env = F12026Env(track_length=5000)

# 2. SAC Agent — new or resumed
resumed_steps = 0
if args.resume:
    ckpt_path, ckpt_steps = find_latest_checkpoint()
    if ckpt_path:
        print(f"Resuming from {ckpt_path} ({ckpt_steps:,} steps)")
        model = SAC.load(ckpt_path, env=env)
        # Load replay buffer if it exists
        buf_path = ckpt_path.replace(".zip", "_replay_buffer.pkl")
        if os.path.exists(buf_path):
            model.load_replay_buffer(buf_path)
            print(f"  Replay buffer loaded ({model.replay_buffer.size():,} transitions)")
        else:
            print("  No replay buffer found, starting with empty buffer")
        resumed_steps = ckpt_steps
    else:
        print("No checkpoint found, starting fresh")
        args.resume = False

if not args.resume:
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=300_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=10_000,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
    )

remaining_steps = TOTAL_TIMESTEPS - resumed_steps
if remaining_steps <= 0:
    print(f"Already at {resumed_steps:,} steps (target: {TOTAL_TIMESTEPS:,}). Nothing to do.")
    print("Increase --total-steps to train further.")
else:
    # 3. Callbacks (save replay buffer with checkpoints for resume)
    callbacks = [
        TqdmCallback(remaining_steps),
        EvalCallback(
            eval_env,
            eval_freq=10_000,
            n_eval_episodes=5,
            best_model_save_path="./best_model/",
            log_path="./eval_logs/",
            deterministic=True,
            verbose=0,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path="./checkpoints/",
            name_prefix="sac_f1",
            save_replay_buffer=True,
        ),
        EvalGifCallback(gif_env, gif_dir="eval_gifs", gif_freq=GIF_FREQ),
    ]

    # 4. Train
    print(f"Training SAC — {remaining_steps:,} steps remaining "
          f"({resumed_steps:,} done / {TOTAL_TIMESTEPS:,} total)")
    print(f"Eval GIF every {GIF_FREQ // 1000}k steps")
    print("Ctrl+C safe — resume with: python3 train_f1.py --resume")
    model.learn(total_timesteps=remaining_steps, callback=callbacks, reset_num_timesteps=False)
    model.save("sac_f1_2026_2d")
    print("\nModel saved as sac_f1_2026_2d.zip")

# 5. Final evaluation
print("\nFinal Evaluation Run...")
obs, _ = eval_env.reset()
history = []
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    x, y, psi = eval_env.unwrapped.get_car_pose()
    history.append({
        'x': x, 'y': y, 'psi': psi,
        'vel': info['velocity_kmh'],
        'soc': info['soc'],
        'time': info['time'],
        'mguk': action[3],
        'steering': action[0],
        'grip': info['grip_ratio'],
        'lateral': info['lateral_offset'],
    })
    done = terminated or truncated

print(f"Lap: {info['time']:.1f}s, Complete: {info.get('lap_complete', False)}")

# 6. Final strategy plot
time_arr = [h['time'] for h in history]
vel = [h['vel'] for h in history]
soc = [h['soc'] for h in history]
mguk = [h['mguk'] for h in history]
steering = [h['steering'] for h in history]
grip = [h['grip'] for h in history]

fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

axes[0].plot(time_arr, vel, color='blue', linewidth=0.8)
axes[0].set_ylabel('Speed (km/h)')
axes[0].set_title('F1 2026 Energy Management Strategy (SAC, 2D)')

axes[1].plot(time_arr, soc, color='green', linewidth=0.8, label='SoC')
axes[1].plot(time_arr, mguk, color='red', alpha=0.4, linewidth=0.8, label='MGU-K')
axes[1].set_ylabel('SoC / MGU-K Action')
axes[1].legend(loc='upper right')

axes[2].plot(time_arr, steering, color='orange', linewidth=0.8)
axes[2].set_ylabel('Steering')
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.3)

axes[3].plot(time_arr, grip, color='purple', linewidth=0.8)
axes[3].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Grip limit')
axes[3].set_ylabel('Grip Ratio')
axes[3].set_xlabel('Time (s)')
axes[3].legend(loc='upper right')

plt.tight_layout()
plt.savefig('f1_2026_strategy.png', dpi=150)
print("Strategy plot saved as 'f1_2026_strategy.png'")
