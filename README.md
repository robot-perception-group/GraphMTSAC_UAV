# GraphMTSAC_UAV

Multi-task reinforcement learning agent for simulated and real-world quadrotor control using [Isaac Gym](https://developer.nvidia.com/isaac-gym) and ArduPilot.

---

## 📄 License

Unless otherwise stated in local licenses or file headers, all code in this repository is:

**Copyright 2024** Max Planck Institute for Intelligent Systems  
Licensed under the terms of the **GNU General Public License v3.0 or later**  
📜 https://www.gnu.org/licenses/gpl-3.0.en.html

---

## 🛠 Installation Guide

### 1. Create a workspace

```bash
mkdir MultitaskRL && cd MultitaskRL
```

---

### 2. Install Isaac Gym (Preview 4)

1. Download Isaac Gym Preview 4 from:  
   👉 https://developer.nvidia.com/isaac-gym

2. Extract it into your workspace:

```bash
tar -xvf IsaacGym_Preview_4_Package.tar.gz
```

3. Run the setup script to install the Isaac Gym conda environment:

```bash
bash IsaacGym_Preview_4_Package/isaacgym/create_conda_env_rlgpu.sh
```

4. Activate the environment:

```bash
conda activate rlgpu
```

> 💡 **Tip:** You may need to edit the script to change the Python version to `3.8` for compatibility with PyTorch 2.0.

---

### 3. Clone this repository and install dependencies

```bash
git clone https://github.com/robot-perception-group/GraphMTSAC_UAV.git
```

Update the Isaac Gym environment with this project's dependencies:

```bash
conda env update --file GraphMTSAC_UAV/environment.yml
```

If needed, install any remaining pip dependencies manually:
```bash
pip install -e GraphMTSAC_UAV
pip install -r GraphMTSAC_UAV/requirements.txt
```

> ✅ This will install all required Python packages, including `wandb`, `gym`, and any others listed in `requirements.txt`.

---

## 🚀 Running the Agent

1. Enter the project directory:

```bash
cd GraphMTSAC_UAV/
```

2. Start training the agent (available options: `SAC`, `MTSAC`, `RMAMTSAC`):

```bash
python run.py agent=MTSAC wandb_log=False env=Quadcopter env.num_envs=25 env.sim.headless=False agent.save_model=False
```

3. Sweep example using [Weights & Biases](https://wandb.ai):

```bash
wandb sweep sweep/mtsac_hyper.yml
```

> Experiment results and logs are saved under the `sweep/` directory and visualized via wandb.

---

## 🌍 Real-World Deployment with ArduPilot

1. Install ArduPilot firmware (see: https://ardupilot.org/dev/docs/building-setup-linux.html)

2. Configure the autopilot to use your custom controller:  
👉 https://ardupilot.org/dev/docs/copter-adding-custom-controller.html

3. Generate the model parameter header from your trained neural network:

```bash
python3 script/cpp_generator_rmagraphnet.py
```

This will generate the file `NN_Parameters.h`.

4. Move the header into the custom ArduPilot folder:

```bash
mv NN_Parameters.h AC_CustomControl/NN_Parameters.h
```

5. Replace ArduPilot's original `AC_CustomControl` directory with this modified one.

6. Compile ArduPilot with the custom controller:

```bash
./waf configure --board Pixhawk6X copter --enable-custom-controller
```


---

## 📂 Project Structure

```
GraphMTSAC_UAV/
├── agents/              # SAC, MTSAC, RMAMTSAC agents
├── assets/              # Quadcopter urdf definition
├── cfg/                 # Training configuration
├── env/                 # Quadcopter simulator settings
├── common/              # Shared utilities, wrappers, layers
├── sweep/               # W&B hyperparameter sweep configs
├── script/              # Tools for real-world deployment
├── AC_CustomControl/    # Custom firmware integration
├── run.py               # Main entry point
├── play.py              # Main testing point
├── requirements.txt
├── environment.yml
```

## Cite (The IROS Verion Comes Later)
```
@article{liu2025multitask,
  title={Multitask Reinforcement Learning for Quadcopter Attitude Stabilization and Tracking using Graph Policy},
  author={Liu, Yu Tang and Vale, Afonso and Ahmad, Aamir and Ventura, Rodrigo and Basiri, Meysam},
  journal={arXiv preprint arXiv:2503.08259},
  year={2025}
}
```