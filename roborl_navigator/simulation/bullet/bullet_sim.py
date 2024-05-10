from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

from roborl_navigator.simulation import Simulation


class BulletSim(Simulation):

    def __init__(
        self,
        render_mode: Optional[str] = "rgb_array",
        n_substeps: Optional[int] = 20,
        renderer: Optional[str] = "Tiny",
        orientation_task: Optional[bool] = False,
        debug_mode: bool = False
    ) -> None:
        super().__init__(render_mode, n_substeps)

        self.orientation_task = orientation_task
        self.background_color = np.array([61.0, 61.0, 61.0]).astype(np.float32) / 255
        options = "--background_color_red={} --background_color_green={} --background_color_blue={}".format(
            *self.background_color
        )
        if self.render_mode == "human":
            self.connection_mode = p.GUI
        elif self.render_mode == "rgb_array":
            if renderer == "OpenGL":
                self.connection_mode = p.GUI
            elif renderer == "Tiny":
                self.connection_mode = p.DIRECT
            else:
                raise ValueError("The 'renderer' argument is must be in {'Tiny', 'OpenGL'}")
        else:
            raise ValueError("The 'render' argument is must be in {'rgb_array', 'human'}")

        self.debug_mode = debug_mode

        self.physics_client = bc.BulletClient(connection_mode=self.connection_mode, options=options)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, int(self.debug_mode))
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, int(self.debug_mode))

        self.n_substeps = n_substeps
        self.timestep = 1.0 / 500
        self.physics_client.setTimeStep(self.timestep)
        self.physics_client.resetSimulation()
        self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physics_client.setGravity(0, 0, -9.81)
        self._bodies_idx = {}

        self.robot_body_name = "panda"
        self.robot_camera_link = 8
        self.camera_pos_local_offset = np.array([0.05, 0.0, 0.02])
        self.image_resolution_width = 128
        self.image_resolution_height = 72
        self.curr_euclid_dist = -1

    def step(self) -> None:
        """Step the simulation."""
        for _ in range(self.n_substeps):
            self.physics_client.stepSimulation()

    def close(self) -> None:
        """Close the simulation."""
        if self.physics_client.isConnected():
            self.physics_client.disconnect()

    def take_image(self):
        camera_pos = self.get_link_position(self.robot_body_name, self.robot_camera_link)
        camera_ori = self.get_link_orientation(self.robot_body_name, self.robot_camera_link)
        camera_pos = list(camera_pos)

        view_matrix = self.get_view_matrix(camera_pos, camera_ori)

        proj_matrix = self.get_proj_matrix()

        return (self.physics_client.getCameraImage(width=self.image_resolution_width,
                                                   height=self.image_resolution_height,
                                                   viewMatrix=view_matrix,
                                                   projectionMatrix=proj_matrix,
                                                   renderer=p.ER_BULLET_HARDWARE_OPENGL), view_matrix, proj_matrix, camera_pos)

    def get_view_matrix(self, camera_pos, camera_ori):

        rot_matrix = p.getMatrixFromQuaternion(camera_ori)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        world_offset = np.dot(rot_matrix, self.camera_pos_local_offset)
        camera_pos += world_offset

        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (1, 0, 0)  # y-axis

        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)

        view_matrix = p.computeViewMatrix(camera_pos, camera_pos + 0.1 * camera_vector, up_vector)

        return view_matrix

    def get_proj_matrix(self):
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self.image_resolution_width) / self.image_resolution_height, nearVal=0.001, farVal=10.0
        )

        return proj_matrix

    def get_point_cloud(self, view_matrix, proj_matrix, img):
        # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

        width = self.image_resolution_width
        height = self.image_resolution_height

        depth = img[3]

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        return points

    def return_closest_dist(self, ee_position, points):
        min_dist = 1000
        min_pos = np.zeros(3)
        for i in range(0, len(points), 50):
            dist = np.linalg.norm(ee_position - points[i], axis=-1)

            if dist <= min_dist:
                min_dist = dist
                min_pos = points[i]

        if self.debug_mode:
            self.set_base_pose("contact_point", min_pos, np.array([0, 0, 0, 1]))
            self.set_base_pose("ee_position", ee_position, np.array([0, 0, 0, 1]))
            self.physics_client.addUserDebugLine(ee_position, min_pos, [1, 0, 0], replaceItemUniqueId=self.lineId)

        min_vector_dist = np.abs(ee_position - min_pos)

        return min_vector_dist, min_dist

    def get_closest_dist(self, ee_position):
        img, view_matrix, proj_matrix, camera_pos = self.take_image()

        points = self.get_point_cloud(view_matrix, proj_matrix, img)

        min_vector_dist, min_euclid_dist = self.return_closest_dist(ee_position, points)

        min_euclid_dist = np.array([min_euclid_dist])
        self.curr_euclid_dist = min_euclid_dist[0]

        return min_vector_dist, min_euclid_dist

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 0)
        yield
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 1)

    # Bullet Unique
    def get_link_position(self, body: str, link: int) -> np.ndarray:
        position = self.physics_client.getLinkState(self._bodies_idx[body], link, computeForwardKinematics=True)[0]
        return np.array(position)

    # Bullet Unique
    def get_link_orientation(self, body: str, link: int) -> np.ndarray:
        orientation = self.physics_client.getLinkState(self._bodies_idx[body], link)[1]
        return np.array(orientation)

    # Bullet Unique
    def get_link_velocity(self, body: str, link: int) -> np.ndarray:
        velocity = self.physics_client.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[6]
        return np.array(velocity)

    # Bullet Unique
    def get_joint_angle(self, body: str, joint: int) -> float:
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[0]

    def set_base_pose(self, body: str, position: np.ndarray, orientation: np.ndarray) -> None:
        if len(orientation) == 3:
            orientation = self.physics_client.getQuaternionFromEuler(orientation)
        self.physics_client.resetBasePositionAndOrientation(
            bodyUniqueId=self._bodies_idx[body], posObj=position, ornObj=orientation
        )

    # Bullet Unique
    def set_joint_angles(self, body: str, joints: np.ndarray, angles: np.ndarray) -> None:
        for joint, angle in zip(joints, angles):
            self.set_joint_angle(body=body, joint=joint, angle=angle)

    # Bullet Unique
    def set_joint_angle(self, body: str, joint: int, angle: float) -> None:
        self.physics_client.resetJointState(bodyUniqueId=self._bodies_idx[body], jointIndex=joint, targetValue=angle)

    # Bullet Unique
    def control_joints(self, body: str, joints: np.ndarray, target_angles: np.ndarray, forces: np.ndarray) -> None:
        self.physics_client.setJointMotorControlArray(
            self._bodies_idx[body],
            jointIndices=joints,
            controlMode=self.physics_client.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=forces,
        )

    # Bullet Unique
    def place_camera(self, target_position: np.ndarray, distance: float, yaw: float, pitch: float) -> None:
        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target_position,
        )

    # Bullet Unique
    def loadURDF(self, body_name: str, **kwargs: Any) -> None:
        self._bodies_idx[body_name] = self.physics_client.loadURDF(**kwargs)

    # OBJECT MANAGER
    def create_scene(self) -> None:
        self.create_plane(z_offset=-0.4)
        self.create_table(length=1.3, width=2, height=0.1)
        self.create_sphere(np.zeros(3))
        self.create_obstacles(length=0.05, width=0.05, height=0.1)

        if self.debug_mode:
            self.create_range_visualization()
            self.create_depth_camera_visualization()

        if self.orientation_task:
            self.create_orientation_mark(np.zeros(3))

    def create_range_visualization(self):

        goal_range = 0.2
        x_low = 0.5 - (goal_range / 2)
        y_low = -goal_range / 2
        z_low = 0.05

        x_high = 0.5 + (goal_range / 2)
        y_high = goal_range / 2
        z_high = 0.05

        low_1 = np.array([x_low, y_low, z_low])
        low_2 = np.array([x_low, y_high, z_high])
        low_3 = np.array([x_high, y_low, z_high])
        low_4 = np.array([x_high, y_high, z_low])

        high_1 = np.array([x_high, y_high, z_high])
        high_2 = np.array([x_high, y_low, z_low])
        high_3 = np.array([x_low, y_high, z_low])
        high_4 = np.array([x_low, y_low, z_high])

        self.physics_client.addUserDebugLine(low_1, high_2, [1, 0, 0])
        self.physics_client.addUserDebugLine(low_1, high_3, [1, 0, 0])
        self.physics_client.addUserDebugLine(low_1, high_4, [1, 0, 0])

        self.physics_client.addUserDebugLine(high_1, low_2, [1, 0, 0])
        self.physics_client.addUserDebugLine(high_1, low_3, [1, 0, 0])
        self.physics_client.addUserDebugLine(high_1, low_4, [1, 0, 0])

        self.physics_client.addUserDebugLine(high_2, low_3, [1, 0, 0])
        self.physics_client.addUserDebugLine(high_3, low_2, [1, 0, 0])

        self.physics_client.addUserDebugLine(high_4, low_3, [1, 0, 0])
        self.physics_client.addUserDebugLine(high_3, low_4, [1, 0, 0])

        self.physics_client.addUserDebugLine(high_2, low_4, [1, 0, 0])
        self.physics_client.addUserDebugLine(high_4, low_2, [1, 0, 0])

    def create_depth_camera_visualization(self) -> None:
        contact_point_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[0, 0, 1, 0.75],
                                                         radius=0.01)
        self._bodies_idx["contact_point"] = self.physics_client.createMultiBody(
            baseVisualShapeIndex=contact_point_visual_shape,
            baseCollisionShapeIndex=-1,
            baseMass=0.0,
            basePosition=np.zeros(3),
        )

        ee_position_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[1, 0, 0, 0.75], radius=0.01)

        self._bodies_idx["ee_position"] = self.physics_client.createMultiBody(
            baseVisualShapeIndex=ee_position_visual_shape,
            baseCollisionShapeIndex=-1,
            baseMass=0.0,
            basePosition=np.zeros(3)
        )
        self.lineId = self.physics_client.addUserDebugLine(np.zeros(3), np.zeros(3))

    def remove_model(self, body_name):
        self.physics_client.removeBody(self._bodies_idx[body_name])

    def create_geometry(
        self,
        body_name: str,
        geom_type: int,
        mass: float = 0.0,
        position: Optional[np.ndarray] = None,
        ghost: bool = False,
        visual_kwargs: Dict[str, Any] = {},
        collision_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Create a geometry."""
        position = position if position is not None else np.zeros(3)
        base_visual_shape_index = self.physics_client.createVisualShape(geom_type, **visual_kwargs)
        if not ghost:
            base_collision_shape_index = self.physics_client.createCollisionShape(geom_type, **collision_kwargs)
        else:
            base_collision_shape_index = -1
        self._bodies_idx[body_name] = self.physics_client.createMultiBody(
            baseVisualShapeIndex=base_visual_shape_index,
            baseCollisionShapeIndex=base_collision_shape_index,
            baseMass=mass,
            basePosition=position,
        )

    def create_box(
        self,
        body_name: str,
        half_extents: np.ndarray,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = None,
    ) -> None:
        rgba_color = rgba_color if rgba_color is not None else np.zeros(4)
        specular_color = np.zeros(3)
        visual_kwargs = {
            "halfExtents": half_extents,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"halfExtents": half_extents}
        self.create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_BOX,
            mass=0.0,
            position=position,
            ghost=False,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    # Bullet Unique
    def create_plane(self, z_offset: float) -> None:
        """Create a plane. (Actually, it is a thin box.)"""
        self.create_box(
            body_name="plane",
            half_extents=np.array([3.0, 3.0, 0.01]),
            position=np.array([0.0, 0.0, z_offset - 0.01]),
            rgba_color=np.array([0.15, 0.15, 0.15, 1.0]),
        )

    # Bullet Unique
    def create_table(self, length: float, width: float, height: float) -> None:
        """Create a fixed table. Top is z=0, centered in y."""
        self.create_box(
            body_name="table",
            half_extents=np.array([length, width, height]) / 2,
            position=np.array([length / 2, 0.0, -height / 2]),
            rgba_color=np.array([0.95, 0.95, 0.95, 1]),
        )

    def create_obstacles(self, length: float, width: float, height: float) -> None:
        self.create_box(
            body_name="obstacle1",
            half_extents=np.array([length, width, height]) / 2,
            position=np.array([0.45, 0.0, height/2]),
            rgba_color=np.array([1, 0, 0, 1]),
        )

        self.create_box(
            body_name="obstacle2",
            half_extents=np.array([length, width, height]) / 2,
            position=np.array([0.45, 0.0, height / 2]),
            rgba_color=np.array([1, 0, 0, 1]),
        )

        self.create_box(
            body_name="obstacle3",
            half_extents=np.array([length, width, height]) / 2,
            position=np.array([0.45, 0.0, height / 2]),
            rgba_color=np.array([1, 0, 0, 1]),
        )

    def create_sphere(self, position: np.ndarray) -> None:
        """Create a sphere."""
        radius = 0.02
        visual_kwargs = {
            "radius": radius,
            "specularColor": np.zeros(3),
            "rgbaColor": np.array([0.0, 1.0, 0.0, 0.5]),
        }
        self.create_geometry(
            "target",
            geom_type=self.physics_client.GEOM_SPHERE,
            mass=0.0,
            position=position,
            ghost=True,
            visual_kwargs=visual_kwargs,
        )

    def create_orientation_mark(self, position: np.ndarray) -> None:
        radius = 0.008
        visual_kwargs = {
            "radius": radius,
            "length": 0.08,
            "specularColor": np.zeros(3),
            "rgbaColor": np.array([0.1, 0.8, 0.1, 0.8]),
        }
        self.create_geometry(
            "target_orientation_mark",
            geom_type=self.physics_client.GEOM_CYLINDER,
            mass=0.0,
            position=position,
            ghost=True,
            visual_kwargs=visual_kwargs,
        )

    def is_collision(self, margin=0.022):
        ds = self.curr_euclid_dist
        collision = ds < margin
        return collision

