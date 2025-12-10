# 🚗 Self-Driving Car with Reinforcement Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyGame](https://img.shields.io/badge/PyGame-2.5.0-green.svg)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An interactive reinforcement learning project where an AI learns to drive a car on a custom track using PPO (Proximal Policy Optimization)**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [Fine-Tuning](#-fine-tuning-guide)

</div>

---

## 🎯 Features

- **🤖 State-of-the-Art RL**: Uses PPO algorithm from Stable-Baselines3
- **👁️ 5 Distance Sensors**: Car uses radar-like sensors to perceive its environment
- **🎮 Interactive Training**: Start/Stop buttons for controlled training sessions
- **📊 Real-Time Visualization**: Watch the AI learn in real-time with live stats
- **🏆 Performance Tracking**: Monitors best distance, episode count, and training progress
- **💾 Model Saving**: Automatically saves trained models for later use
- **🎨 Custom Track Designer**: Easy-to-modify track layouts

---

## 🎬 Demo

### Training Progress
The car starts with no knowledge and gradually learns to navigate the track:

**Episode 1:** 💥 Immediate crash
**Episode 50:** 🚗 Basic steering
**Episode 200:** 🏎️ Smooth cornering
**Episode 500+:** 🏁 Professional driving

### Stats Display
```
Episode: 523
Steps: 45,231
Distance: 2,847
Best: 3,102
Status: TRAINING
```

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB RAM minimum
- Graphics display (for visualization)

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rl-self-driving-car.git
cd rl-self-driving-car
```

### 2. Install Dependencies
```bash
pip install pygame numpy gymnasium stable-baselines3
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 3. Run the Program
```bash
python rl_car.py
```

---

## 🎮 Usage

### Starting Training

1. **Launch the program**
   ```bash
   python rl_car.py
   ```

2. **Click the START button** (green) to begin training

3. **Watch the AI learn!** The car will:
   - Start crashing immediately
   - Learn to avoid walls
   - Develop smooth steering
   - Master the track

4. **Click STOP** (red) to pause training at any time

5. **Resume anytime** by clicking START again

### Keyboard Shortcuts
- `ESC` or close window to exit
- Training auto-saves on interruption (Ctrl+C)

---

## 🧠 How It Works

### Architecture Overview

```
┌─────────────────┐
│  5 Radar Sensors│ ──► Measures distance to walls
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Neural Network │ ──► PPO Policy (2-layer MLP)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   3 Actions     │ ──► Turn Left / Straight / Turn Right
└─────────────────┘
```

### Sensors
The car has **5 distance sensors** positioned at:
- `-90°` (Left)
- `-45°` (Front-Left)
- `0°` (Front)
- `+45°` (Front-Right)
- `+90°` (Right)

Each sensor measures distance to the nearest wall (normalized 0-1).

### Action Space
```python
0 = Turn Left (7° rotation)
1 = Go Straight (no rotation)
2 = Turn Right (7° rotation)
```

### Reward System
```python
# Staying alive
+1.0 per frame

# Distance traveled
+0.01 per unit distance

# Safe driving
+0.5 if maintaining good distance from walls

# Smooth driving
-0.1 penalty for turning (encourages straight driving)

# Crashing
-50 + (0.1 × distance_traveled)
```

### PPO Hyperparameters
```python
learning_rate = 3e-4      # Step size for weight updates
n_steps = 2048            # Steps before policy update
batch_size = 64           # Training batch size
n_epochs = 10             # Optimization passes per update
gamma = 0.99              # Discount factor
gae_lambda = 0.95         # Advantage estimation
clip_range = 0.2          # PPO clipping parameter
ent_coef = 0.01          # Entropy bonus
```

---

## 🎛️ Fine-Tuning Guide

### Adjusting Learning Speed

**Faster Learning** (less stable):
```python
learning_rate = 1e-3  # More aggressive updates
```

**Slower Learning** (more stable):
```python
learning_rate = 1e-4  # Conservative updates
```

**Balanced** (recommended):
```python
learning_rate = 3e-4  # Industry standard
```

### Modifying the Track

Edit `draw_track()` in the `TrackEnv` class:
```python
def draw_track(self):
    points = [
        (x1, y1), (x2, y2), (x3, y3), ...  # Your custom points
    ]
    pygame.draw.lines(self.map_surface, ROAD_COLOR, True, points, 120)
    #                                                              ^^^ track width
```

### Customizing Rewards

In the `step()` method:
```python
# Increase forward progress reward
reward += self.car.distance_traveled * 0.05  # Was 0.01

# Reduce crash penalty
reward = -25 + (self.car.distance_traveled * 0.1)  # Was -50

# Add speed bonus
reward += self.car.speed * 0.1
```

### Training Longer
```python
model.learn(total_timesteps=500000)  # Default is 100,000
```

---

## 📊 Monitoring Training

### Using TensorBoard

The project automatically logs training metrics to TensorBoard:

```bash
# In a separate terminal
tensorboard --logdir=./ppo_car_tensorboard/

# Open browser to: http://localhost:6006
```

**Metrics tracked:**
- Episode reward
- Episode length
- Policy loss
- Value loss
- Learning rate

### Console Output
```
Episode finished - Reward: 245.32, Length: 387
Episode finished - Reward: 512.18, Length: 891
Episode finished - Reward: 1023.45, Length: 1456
```

---

## 💾 Saving & Loading Models

### Auto-Save
Models automatically save on:
- Training completion
- Keyboard interrupt (Ctrl+C)

### Manual Save
```python
model.save("my_trained_car")
```

### Loading a Trained Model
```python
from stable_baselines3 import PPO

model = PPO.load("ppo_self_driving_car")

# Test the trained model
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()
```

---

## 🔧 Configuration

### Display Settings
```python
WIDTH, HEIGHT = 1200, 800  # Window size
FPS = 60                   # Frame rate
SENSOR_LENGTH = 200        # Radar range
```

### Car Settings
```python
CAR_SIZE = 20              # Car dimensions
speed = 5                  # Movement speed
angle = 0                  # Starting rotation
```

### Colors
```python
ROAD_COLOR = (100, 100, 100)  # Gray
CAR_COLOR = (255, 0, 0)       # Red
SENSOR_COLOR = (0, 255, 0)    # Green
```

---

## 🐛 Troubleshooting

### Issue: Car crashes immediately every episode
**Solution:** 
- Increase `SENSOR_LENGTH` to 250-300
- Reduce `speed` to 3-4
- Widen the track (increase width parameter)

### Issue: Training is too slow
**Solution:**
- Increase `learning_rate` to 1e-3
- Reduce `n_steps` to 1024
- Decrease FPS to 30 for faster simulation

### Issue: Car learns then forgets
**Solution:**
- Decrease `learning_rate` to 1e-4
- Increase `batch_size` to 128
- Check reward function for conflicting signals

### Issue: "pygame not found" error
**Solution:**
```bash
pip install pygame --upgrade
```

---

## 🎓 Learning Resources

### Understanding Reinforcement Learning
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Algorithm Explained](https://openai.com/blog/openai-baselines-ppo/)
- [Deep RL Course by Hugging Face](https://huggingface.co/deep-rl-course)

### Advanced Topics
- **Curriculum Learning**: Gradually increase track difficulty
- **Transfer Learning**: Train on simple tracks, test on complex ones
- **Multi-Agent**: Multiple cars learning simultaneously

---

## 🗺️ Roadmap

- [ ] Add multiple track layouts
- [ ] Implement curriculum learning
- [ ] Add manual control mode
- [ ] Create track editor UI
- [ ] Add obstacles and dynamic elements
- [ ] Support for continuous action space (SAC/TD3)
- [ ] Multiplayer racing mode
- [ ] 3D visualization option

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Stable-Baselines3** team for the excellent RL library
- **OpenAI** for PPO algorithm research
- **Pygame** community for the graphics framework
- Inspired by various self-driving car simulations

---

## 📧 Contact

**Your Name** - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/rl-self-driving-car](https://github.com/yourusername/rl-self-driving-car)

---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**

Made with ❤️ and 🤖 by [Your Name]

</div>
