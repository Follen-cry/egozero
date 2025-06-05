<p align="center">
    <img src="assets/egozero_github.gif" width="70%">
</p>


<h1 align="center" style="font-size: 2.0em; font-weight: bold; margin-bottom: 0; border: none; border-bottom: none;">EgoZero:<br>Robot Learning from Smart Glasses</h1>

##### <p align="center">[Vincent Liu<sup>*1</sup>](https://vliu15.github.io)&emsp;[Ademi Adeniji<sup>*12</sup>](https://ademiadeniji.github.io/)&emsp;[David Zhan<sup>*1</sup>](https://linkedin.com/in/david-zhan-96935126a)</p>
##### <p align="center">[Siddhant Haldar<sup>1</sup>](https://siddhanthaldar.github.io/)&emsp;[Raunaq Bhirangi<sup>1</sup>](https://raunaqbhirangi.github.io/)&emsp;[Pieter Abbeel<sup>2</sup>](https://people.eecs.berkeley.edu/~pabbeel/)&emsp;[Lerrel Pinto<sup>1</sup>](https://lerrelpinto.com)</p>
##### <p align="center"><sup>1</sup>New York University&emsp;<sup>2</sup>UC Berkeley</p>
##### <p align="center"><sup>*</sup>Equal contribution</p>

<div align="center">
    <a href="https://arxiv.org/abs/2505.20290"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red"></a> &ensp;
    <a href="https://egozero-robot.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Website&color=blue"></a> &ensp;
</div>

## table of contents
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Franka-Teach](#franka-teach)

## installation
```bash
git submodule update --init --recursive
conda create -y -n egozero python=3.10
conda activate egozero
bash setup.sh
```

Add the following environment variables to your `~/.bashrc` or `~/.zshrc` with your institution's username and password
```bash
export ARIA_MPS_UNAME="your_uname"
export ARIA_MPS_PASSW="your_passw"
```

Verify that your dependencies have been installed correctly, run
```bash
aria-doctor
```

Pair the glasses via USB to your computer
```bash
aria auth pair
```

## data collection
To record offline so that the complete `.vrs` file can be submitted to the MPS server for data postprocessing, you must first install the [Aria mobile app](https://facebookresearch.github.io/projectaria_tools/docs/ARK/mobile_companion_app) and the [Aria studio](https://facebookresearch.github.io/projectaria_tools/docs/ARK/aria_studio). Then,
1. Connect to the glasses on the Aria mobile app
2. Create a new recording session
3. Transfer the `.vrs` file onto your computer

Submit the video for data processing on the MPS server and reorganize the output folder. Job submission may take anywhere from 5 to 30 minutes. For example, if your `.vrs` file is `pick_bread_1.vrs`, you would run
```bash
bash scripts/submit_mps.sh pick_bread_1
```

> In our experiments, we collect all our data with only the right hand and reset the task with the left hand. You may swap this, but our preprocessing script segments individual demonstrations based on absence of the specified hand.

## preprocessing
Copy the collect data to this repo on a machine where you will run preprocessing. Ideally this machine has GPU compute.

Label the expert points on your demonstration by opening `label_points.ipynb` with a Python kernel. Modify the paths in the first cell and run the entire notebook. Label points on the displayed image by clicking points and click the `Save Points` button. Run full preprocessing with
```bash
python preprocess.py --mps_sample_path mps_pick_bread_1_vrs/ --is_right_hand --prompts "a bread slice." "a plate."
```

## training
1. Create a new config yaml for your new task at `point_policy/cfgs/suite/task/franka_env/` and customize the `num_object_points`, `root_dir`, and `prompts` fields. See `point_policy/cfgs/suite/task/franka_env/pick_bread.yaml` for reference.
2. Modify `scripts/train.sh` to point to your new dataset and task config. Set the `data_dirs` and `experiment` variables.
3. Train the model with `bash scripts/train.sh`. See `point_policy/cfgs/config.yaml` and `point_policy/cfgs/suite/aria.yaml` for hydra flags from command line.

## inference
First go through the [Franka-Teach](#franka-teach) section to make sure the hardware is running correctly.
1. Modify `scripts/eval.sh` to point to your new dataset, task config, and checkpoint weights (should be saved in `point_policy/exp_local`).
2. Inference the model with `bash scripts/eval.sh`. See `point_policy/cfgs/config.yaml` and `point_policy/cfgs/suite/aria.yaml` for hydra flags from command line.

To stream the iPhone to get RGBD for robot rollout, run
```bash
python scripts/stream_iphone.py
```

## franka-teach
To run the robot, see the [Franka-Teach](https://github.com/NYU-robot-learning/Franka-Teach) repository for how to run Franka robots
