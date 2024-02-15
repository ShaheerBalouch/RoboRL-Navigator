import os
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
        self.physics_client = bc.BulletClient(connection_mode=self.connection_mode, options=options)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        self.n_substeps = n_substeps
        self.timestep = 1.0 / 500
        self.physics_client.setTimeStep(self.timestep)
        self.physics_client.resetSimulation()
        self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physics_client.setGravity(0, 0, -9.81)
        self._bodies_idx = {}

    def step(self) -> None:
        """Step the simulation."""
        for _ in range(self.n_substeps):
            self.physics_client.stepSimulation()

    def close(self) -> None:
        """Close the simulation."""
        if self.physics_client.isConnected():
            self.physics_client.disconnect()

    def render(
        self,
        width: int = 720,
        height: int = 480,
        target_position: Optional[np.ndarray] = None,
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            target_position = target_position if target_position is not None else np.zeros(3)
            view_matrix = self.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2,
            )
            proj_matrix = self.physics_client.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
            )
            (_, _, rgba, _, _) = self.physics_client.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                shadow=True,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            rgba = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
            return rgba[..., :3]

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 0)
        yield
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 1)

    # Bullet Unique
    def get_link_position(self, body: str, link: int) -> np.ndarray:
        position = self.physics_client.getLinkState(self._bodies_idx[body], link)[0]
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
        if self.orientation_task:
            self.create_orientation_mark(np.zeros(3))

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
        baseVisualShapeIndex = self.physics_client.createVisualShape(geom_type, **visual_kwargs)
        if not ghost:
            baseCollisionShapeIndex = self.physics_client.createCollisionShape(geom_type, **collision_kwargs)
        else:
            baseCollisionShapeIndex = -1
        self._bodies_idx[body_name] = self.physics_client.createMultiBody(
            baseVisualShapeIndex=baseVisualShapeIndex,
            baseCollisionShapeIndex=baseCollisionShapeIndex,
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

    def create_sphere(self, position: np.ndarray) -> None:
        """Create a sphere."""
        radius = 0.02
        visual_kwargs = {
            "radius": radius,
            "specularColor": np.zeros(3),
            "rgbaColor": np.array([0.9, 0.1, 0.1, 0.75]),
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
        # Target Box Orientation Lines for Debug
        # oid = self._bodies_idx["panda"]
        # line_length = 30  # Length of the lines
        # p.addUserDebugLine([0, 0, 0], [line_length, 0, 0], [1, 0, 0], parentObjectUniqueId=oid, parentLinkIndex=10)
        # p.addUserDebugLine([0, 0, 0], [0, line_length, 0], [0, 1, 0], parentObjectUniqueId=oid, parentLinkIndex=10)
        # p.addUserDebugLine([0, 0, 0], [0, 0, line_length], [0, 0, 1], parentObjectUniqueId=oid, parentLinkIndex=10)
