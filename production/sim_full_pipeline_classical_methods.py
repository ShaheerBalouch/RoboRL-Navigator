from production.ros_controller import ROSController

ros_controller = ROSController()

# Open the gripper
ros_controller.hand_open()
# Go to Image Capturing Location
ros_controller.go_to_capture_location()
# Save image, depth data and camera info
ros_controller.capture_image_and_save_info()
# View image
ros_controller.view_image()

# Send Request to Contact Graspnet Server
ros_controller.request_graspnet_result()
# Parse Responded File
raw_pose = ros_controller.process_grasping_results()
# Transform Frame to Panda Base
pose = ros_controller.transform_camera_to_world(raw_pose)
ros_controller.get_pose_goal_plan_with_duration(pose, 'rrt')
# Go to target (grasping) pose
ros_controller.go_to_pose_goal(pose)

ros_controller.hand_close()
