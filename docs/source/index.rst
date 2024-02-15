Documentation of RoboRL Navigator
=================================

**RoboRL Navigator** is a project that offers a codebase for Reinforcement Learning tailored to manipulator robots, particularly the Franka Emika Panda Robot. The project includes both Bullet and ROS Gazebo simulation environments that can be used to train the model to reach a specified pose. It also utilizes an open-source Grasping Pose Detection project that can be tested in either the Gazebo Simulation environment or the Real World.

Explore the :doc:`usage` section for more details.



.. toctree::
   :maxdepth: 2
   :caption: Install

   installation
   validate-installation

   :maxdepth: 2
   :caption: Train
   environments
   train-your-model
   download-trained-model

   :maxdepth: 2
   :caption: Test
   model-evaluation
   grasping-pose-detection

   :maxdepth: 2
   :caption: Demonstrate
   simulation-demonstration
   real-world-demonstration
