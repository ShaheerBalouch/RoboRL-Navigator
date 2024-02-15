Grasping Pose Detection (GPD)
=============================

.. code:: shell

   conda activate contact_graspnet_env
   python contact_graspnet/contact_graspnet_server.py

View Grasping Poses
-------------------

.. code:: shell

    # make sure conda env activated
    python3 contact_graspnet/inference.py \
        --np_path=/path/to/your/data.npy \
        --forward_passes=5 \
        --z_range=[0.2,1.1]


GPD Server
----------

Run GPD server

.. code:: shell

    python3 contact_graspnet/contact_graspnet_server.py


API Reference
-------------


**Local Configuration**


.. code:: http

     GET /run

Request

+-----------+------+-------------+
| Parameter | Type | Description |
+===========+======+=============+
| ``file``  | FILE | **Required**|
+-----------+------+-------------+

Response

+-----------+------+-------------+
| Parameter | Type | Description |
+===========+======+=============+
| ``file``  | FILE | **Required**|
+-----------+------+-------------+

**LAN Configuration**

.. code:: http

     GET /run


Request

+-----------+------+-------------+
| Parameter | Type | Description |
+===========+======+=============+
| ``file``  | FILE | **Required**|
+-----------+------+-------------+

Response

+-----------+------+-------------+
| Parameter | Type | Description |
+===========+======+=============+
| ``file``  | FILE | **Required**|
+-----------+------+-------------+

