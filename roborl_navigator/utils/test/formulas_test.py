import numpy as np
import unittest

from pybullet import getQuaternionFromEuler, getEulerFromQuaternion
from roborl_navigator.utils.formulas import euler_to_quaternion, quaternion_to_euler


class TestFormulas(unittest.TestCase):

    def test_quaternion_formulas(self):
        # TOLERANCE IS .001
        euler_orientation = np.array([0.5, 1.5, -2.7])
        pb_quaternion = getQuaternionFromEuler(euler_orientation)
        util_quaternion = euler_to_quaternion(euler_orientation)
        np.testing.assert_allclose(
            pb_quaternion,
            util_quaternion,
            atol=1e-3,
            err_msg="Converted joint state does not match the expected result.",
        )
        pb_euler = getEulerFromQuaternion(util_quaternion)
        util_euler = quaternion_to_euler(pb_quaternion)
        np.testing.assert_allclose(
            euler_orientation, pb_euler, atol=1e-3, err_msg="Converted joint state does not match the expected result."
        )
        np.testing.assert_allclose(
            euler_orientation,
            util_euler,
            atol=1e-3,
            err_msg="Converted joint state does not match the expected result.",
        )


if __name__ == '__main__':
    unittest.main()
