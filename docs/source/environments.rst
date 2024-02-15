Environments
============


Bullet RL Environment
---------------------

Environment Name: ``RoboRL-Navigator-Panda-Bullet``

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Parameter
     - Values
     - Default
     - Description
   * - ``render_mode``
     - ``human, rgb_array``
     - ``human``
     - Rendering mode of the physics engine
   * - ``orientation_task``
     - ``boolean``
     - ``False``
     - Activate orientation features of targets
   * - ``distance_threshold``
     - ``float``
     - ``0.05``
     - Distance tolerance in cm
   * - ``goal_range``
     - ``float``
     - ``0.3``
     - Goal region dimension (x=gr, y=gr, z=gr/2)



ROS RL Environment
------------------

Environment Name: ``RoboRL-Navigator-Panda-ROS``


.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Parameter
     - Values
     - Default
     - Description
   * - ``orientation_task``
     - ``boolean``
     - ``False``
     - Activate orientation features of targets
   * - ``distance_threshold``
     - ``float``
     - ``0.05``
     - Distance tolerance in cm
   * - ``goal_range``
     - ``float``
     - ``0.3``
     - Goal region dimension (x=gr, y=gr, z=gr/2)
   * - ``demonstration``
     - ``boolean``
     - ``False``
     - Enables demonstration properties such as target visualiser and info
   * - ``real_robot``
     - ``boolean``
     - ``False``
     - Updates robot name for integration with real robot





