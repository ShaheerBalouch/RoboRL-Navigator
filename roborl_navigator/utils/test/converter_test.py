import numpy as np
import unittest

from roborl_navigator.utils.converter import PandaConverter


class TestConversion(unittest.TestCase):

    def test_conversion(self):
        panda_converter = PandaConverter()
        initial_joints = np.array([0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78])
        bullet_joints = panda_converter.real_to_bullet(initial_joints)
        reversed_joints = panda_converter.bullet_to_real(bullet_joints)

        np.testing.assert_allclose(
            initial_joints,
            reversed_joints,
            atol=1e-3,
            err_msg="Converted joint state does not match the expected result.",
        )


if __name__ == '__main__':
    unittest.main()
