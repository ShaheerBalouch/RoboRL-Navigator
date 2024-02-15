Installation
============

Suggested Python version is **3.8**.

Repository and Python Environment
---------------------------------

Clone repository

.. code:: shell

   git clone https://github.com/eminsafa/RoboRL-Navigator.git

Create Python environment

.. code:: shell

   cd RoboRL-Navigator
   python3 -m venv roborl-navigator-env
   source roborl-navigator-env/bin/activate

As this project use model-based importing, please make sure you have
added module path as environment variable.

.. code:: shell

   cd RoboRL-Navigator/
   export PYTHONPATH=$PYTHONPATH:$(pwd)

ROS and Gazebo Installation
---------------------------

It is required to install ROS Noetic version. This guide is for Cardiff
University Computational Robotics Team. You can install ROS by following
official instructions `here <http://wiki.ros.org/noetic/Installation>`__.

Install and Build ``tue-env``

.. code:: shell

   source <(wget -O - https://raw.githubusercontent.com/CardiffUniversityComputationalRobotics/tue-env/cucr/installer/bootstrap_cucr.bash)
   cucr-get install cucr-dev
   cucr-make
   source ~/.bashrc

Install Franka Panda related test environment. You may prefer
``robo_rl`` branch that is especially created for this project.

.. code:: shell

   cucr-get install ros-test_franka_simulation_bringup
   cucr-make
   source ~/.bashrc

Install Requirements
--------------------

Install RoboRL Navigator Python requirements

.. code:: shell

   # Basics
   pip3 install -r requirements.txt
   # For ROS installed OS
   pip3 install -r requirements_ros.txt

GraspNet Installation
---------------------

.. code:: shell

   git clone https://github.com/eminsafa/contact_graspnet.git
   cd contact_graspnet
   conda env create -f contact_graspnet_env.yml

Download trained models for Contact Graspnet from
`here <https://drive.google.com/file/d/1tQDtYyQv5-QTuLvvPJLhfdZ6tINGBv-L/view?usp=sharing>`__
and extract files under ``external/contact_graspnet/checkpoints``
directory. It will look like this:

.. code:: shell

   external
   └── contact_graspnet
       └── checkpoints
           └── scene_test_2048_bs3_hor_sigma_001
               ├── checkpoint
               ├── config.yaml
               ...

Instruction to run ROS environment is
`here <running_ros_and_gazebo.md>`__. You can validate your
installations `here <validate_installation.md>`__.
