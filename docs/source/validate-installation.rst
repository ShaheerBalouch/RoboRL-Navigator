Validate Installation
=====================

Bullet Environment validation

.. code:: shell

    python3 test/bullet_env_init.py


If you have successfully run ROS and Gazebo, you can run ROS environment

.. code:: shell

    python3 test/ros_env_init.py


Running ROS and Gazebo
----------------------

Run following command to run ROS with simulation launch file

.. code:: shell

    test-franka-simulation-full camera:=true


After simulation initialised, run RVIZ

.. code:: shell

    test-franka-desktop-rviz
