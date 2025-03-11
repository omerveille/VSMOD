import unittest
from .test_utils import mute_outputs

import slicer
from ransac_slicer.dependencies_checker import (
    install_missing_module,
    check_missing_module_pip,
    check_and_install_missing_dependencies,
    required_modules,
)
from slicer.util import pip_uninstall


class DependenciesInstallationTest(unittest.TestCase):
    def test_01_install_missing_module(self):
        slicer.mrmlScene.Clear()
        self.addCleanup(self.handle_failure)
        with mute_outputs():
            pip_uninstall(
                "numpy scipy trimesh scikit-image networkx numba intel-openmp"
            )
            missing_modules = check_missing_module_pip(required_modules)

            self.assertListEqual(sorted(missing_modules), sorted(required_modules))

            with slicer.util.tryWithErrorDisplay(
                "Failed to install dependencies.", waitCursor=True
            ):
                install_missing_module(missing_modules)

            missing_modules = check_missing_module_pip(required_modules)
            self.assertFalse(missing_modules)

    def handle_failure(self):
        with mute_outputs():
            check_and_install_missing_dependencies()
