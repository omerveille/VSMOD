from __future__ import annotations

from slicer.util import pip_install
from packaging import version

from .popup_utils import make_custom_progress_bar
import slicer
import math
import re
from shutil import which

required_modules = {
    "numpy": version.parse("2.0.2"),
    "scipy": version.parse("1.13.1"),
    "trimesh": version.parse("4.6.5"),
    "scikit-image": version.parse("0.24.0"),
    "networkx": version.parse("3.2.1"),
    "numba": version.parse("0.60.0"),
    "intel-openmp": version.parse("2024.2.1"),
}


def check_missing_module_pip() -> dict[str, version.Version]:
    """
    Check wether a module is installed or not using PythonSlicer's pip.
    It allows us to check dependencies in real-time.

    Parameters
    ----------
    module_names: list[str]
        Name of the modules we check may be installed.

    Returns
    ----------
    dict[str, version.Version]
        A dict of all the missing packages we have to install and their version.
    """
    PythonSlicer_path = which("PythonSlicer")
    command = [PythonSlicer_path, "-m", "pip", "show", *list(required_modules.keys())]
    proc = slicer.util.launchConsoleProcess(command, useStartupEnvironment=False)

    command_output: str = proc.stdout.read()
    found_modules = re.findall(r"Name:\s*(.+)\s*Version:\s*(.+)\s*", command_output)
    found_modules = {
        package_name: version.parse(package_version)
        for package_name, package_version in found_modules
    }

    missing_packages = dict()

    for package_name, package_version in required_modules.items():
        missing = (
            True
            if found_modules.get(package_name) is None
            else found_modules[package_name] < package_version
        )
        if missing:
            missing_packages[package_name] = package_version

    proc.wait()
    return missing_packages


def install_missing_module(missing_modules: dict[str, version.Version]) -> None:
    """
    Check that the module is installed, if not install it.

    Parameters
    ----------
    missing_modules: dict[str, version.Version]
        A dictionnary containing the missing modules and their associated needed version
    """
    progress_bar = make_custom_progress_bar(
        labelText="Installing dependency...",
        windowTitle="Installing dependencies...",
        width=300,
    )
    print("Installing missing dependencies...")

    missing_modules_list = [
        (f"{module_name}>={module_version}", module_name)
        for module_name, module_version in missing_modules.items()
    ]
    for i, module in enumerate(missing_modules_list):
        command, module_name = module
        install_text = f"Installing {module_name}..."
        print(install_text)
        progress_bar.labelText = install_text
        slicer.app.processEvents()

        pip_install(command)
        install_text = f"{module_name} installed !"
        print(install_text)
        progress_bar.labelText = install_text
        progress_bar.value = math.floor(((i + 1) / len(missing_modules_list)) * 100)
        slicer.app.processEvents()

    progress_bar.close()


def check_and_install_missing_dependencies() -> bool:
    """
    Check if the user has all the required dependencies.
    If not, install them.

    Returns
    -------

    A flag to tell if a restart is needed or not.
    """

    missing_modules = check_missing_module_pip()

    if missing_modules:
        with slicer.util.tryWithErrorDisplay(
            "Failed to install dependencies.", waitCursor=True
        ):
            install_missing_module(missing_modules)
            return True

    return False
