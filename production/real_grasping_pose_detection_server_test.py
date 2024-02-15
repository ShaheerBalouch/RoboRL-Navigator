import numpy as np
import time
from ros_controller import ROSController

remote_ip = "http://172.20.10.10:6161/run"

ros_controller = ROSController(real_robot=True)

ros_controller.go_to_capture_location()

for i in range(10):
    input("continue?")
    ros_controller.capture_image_and_save_info()
    ros_controller.request_graspnet_result()
    saved_file_path = ros_controller.request_graspnet_result(remote_ip=remote_ip)
    # Parse Responded File
    target_pose_by_camera = ros_controller.process_grasping_results(path=saved_file_path)
