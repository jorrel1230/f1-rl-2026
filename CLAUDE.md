# F1 2026 RL Energy Management

## Project
RL agent learns optimal racing line + energy strategy under 2026 F1 regulations.
Key thesis: optimal racing line changes under 2026 regs (3x MGU-K regen, no MGU-H → braking zones are energy goldmines → tighter corner entries may be optimal).

Princeton COS 435/ECE 433 group project.

## Architecture

### `f1_2026_env.py` — Gymnasium environment
- **Full 2D bicycle model** with steering, throttle, brake, MGU-K deploy
- **State**: [x, y, heading, velocity, SoC] (internal)
- **Observations**: 12D normalized — velocity, SoC, lateral offset, heading error, 5-pt curvature lookahead, harvest remaining, yaw rate, progress
- **Actions**: 4D continuous — [steering, throttle, brake, mguk_deploy]
- **Track**: 18-waypoint Barcelona-inspired circuit, periodic cubic spline, 12m width, KDTree closest-point lookup
- **Physics**: friction circle tire model (combined lat+long grip), bicycle kinematics (wheelbase 3.6m)
- **Energy (2026 regs)**: 350kW MGU-K, 4MJ battery, 8.5MJ/lap harvest cap, 768kg, 400kW ICE, 80% regen efficiency
- **Regen**: automatic on any deceleration (brake, coast, lift-off, grip loss) — not just explicit brake action
- **Reward**: progress along track (primary) + milestones at 25/50/75/90% (+20 each) + lap completion (1000/time) - off-track (-50, terminate) - grip exceeded (-5×excess) - lateral offset penalty
- **Episode**: rolling start 30 m/s, max 20,000 steps (1000s), terminates on lap complete or off-track

### `train_f1.py` — Training script
- **SAC** (Soft Actor-Critic) — better than PPO for 4D continuous actions
- net_arch=[256,256], lr=1e-3, buffer=300k, batch=256, learning_starts=10k
- **Callbacks**: tqdm progress, EvalCallback (best model save), CheckpointCallback (every 50k + replay buffer), EvalGifCallback (eval GIF + racing line PNG every 50k)
- **Resume support**: `python train_f1.py --resume` loads latest checkpoint + replay buffer
- Args: `--total-steps`, `--gif-freq`, `--resume`

### `generate_f1_gif.py` — Visualization
- Loads saved model, runs eval episode
- Dark theme, track boundaries, heading arrow, speed-colored trail (plasma cmap)
- Outputs: `f1_2026_lap.gif` + `f1_2026_racing_line.png` (learned line vs centerline)

## Key commands
```bash
# Train from scratch
python train_f1.py

# Resume training
python train_f1.py --resume

# Train more steps
python train_f1.py --resume --total-steps 1000000

# Generate visualization (needs trained model)
python generate_f1_gif.py
```

## Output directories
- `eval_gifs/` — periodic eval GIFs + racing line PNGs during training
- `best_model/` — best model from eval callback
- `checkpoints/` — periodic checkpoints with replay buffers
- `eval_logs/` — evaluation metrics

## Current status
- Env change invalidated prior training (added auto-regen on deceleration, progress milestones, bumped max_steps to 20k, lr to 1e-3)
- Training from scratch needed
- CPU-bound (MlpPolicy, no CNN). Laptop or cluster both fine.

## Dependencies
```
gymnasium>=1.0.0, matplotlib, numpy, pillow, scipy, stable-baselines3>=2.0.0, torch, tqdm
```
Cluster: use conda for torch (`conda create -n f1rl python=3.9 pytorch -c pytorch`), pip for rest.
