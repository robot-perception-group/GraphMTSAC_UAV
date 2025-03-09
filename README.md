=================================================================

# Copyright and License

All Code in this repository - unless otherwise stated in local license or code headers is

Copyright 2024 Max Planck Institute for Intelligent Systems

Licensed under the terms of the GNU General Public Licence (GPL) v3 or higher.
See: https://www.gnu.org/licenses/gpl-3.0.en.html



# installation
- create workspace
```console
mkdir MultitaskRL
cd MultitaskRL
```

- setup isaac-gym 
1. download isaac-gym from https://developer.nvidia.com/isaac-gym
2. extract isaac-gym to the workspace 
3. create conda environment and install the dependencies 
```console
bash IsaacGym_Preview_4_Package/isaacgym/create_conda_env_rlgpu.sh 
conda activate rlgpu
```

- clone this repository to the workspace and install dependencies
```console
git clone https://github.com/robot-perception-group/GraphMTSAC_UAV.git
pip install -r GraphMTSAC_UAV/requirements.txt
```

# Run Agent: 
- enter the RL workspace
```console
cd GraphMTSAC_UAV/
```

- start learning in 25 environments with agents available: SAC, MTSAC, RMAMTSAC
```
python run.py agent=MTSAC wandb_log=False env=Quadcopter env.num_envs=25  env.sim.headless=False agent.save_model=False
```

- The experiments are stored in the sweep folder. For example, hyperparameter tuning for the mtsac agent
```console
wandb sweep sweep/mtsac_hyper.yml
```
The experimental results are gathered in Wandb. 


# Real World Experiment:
- install ardupilot (https://ardupilot.org)

- Modify the model path in 'script/cpp_generator_rmagraphnet.py' and execute firmware generator
```console
python3 script/cpp_generator_rmagraphnet.py
```
this will generate a file 'NN_Parameters.h'.

- Move this file to AC_CustomControl folder
```console
mv NN_Parameters.h AC_CustomControl/NN_Parameters.h
```

- Copy the entire AC_CustomControl folder and replace the Ardupilot AC_CustomControl folder. Use following command to compile ardupilot: 
```console
./waf configure --board Pixhawk6X copter --enable-custom-controller
```

- You will need to configure ardupilot to enable custom control: https://ardupilot.org/dev/docs/copter-adding-custom-controller.html

