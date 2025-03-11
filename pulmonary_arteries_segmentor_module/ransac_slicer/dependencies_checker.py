from __future__ import annotations

from slicer.util import pip_install

from .popup_utils import make_custom_progress_bar
import slicer
import math
import re
from shutil import which

required_modules = [
    "numpy",
    "scipy",
    "trimesh",
    "scikit-image",
    "networkx",
    "numba",
    "intel-openmp",
]


def check_missing_module_pip(module_names: list[str]) -> bool:
    """
    Check wether a module is installed or not using PythonSlicer's pip.
    It allows us to check dependencies in real-time.

    Parameters
    ----------
    module_names: list[str]
    Name of the modules we check may be installed.
    """
    PythonSlicer_path = which("PythonSlicer")
    command = [PythonSlicer_path, "-m", "pip", "show", *module_names]
    proc = slicer.util.launchConsoleProcess(command, useStartupEnvironment=False)

    output_first_line: str = proc.stdout.read().splitlines()[0]

    missing_packages = []
    if output_first_line.startswith("WARNING"):
        match = re.search(r"WARNING: Package\(s\) not found: (.+)", output_first_line)
        missing_packages = [pkg.strip() for pkg in match.group(1).split(",")]

    proc.wait()
    return missing_packages


def install_missing_module(modules: list[str | tuple[str, str]]) -> None:
    """
    Check that the module is installed, if not install it.

    Parameters
    ----------
    modules: list of str or tuple[str, str], modules to install.
    """
    progress_bar = make_custom_progress_bar(
        labelText="Installing dependency...",
        windowTitle="Installing dependencies...",
        width=300,
    )
    print("Installing missing dependencies...")
    for i, module in enumerate(modules):
        module_name = module[1] if isinstance(module, tuple) else module
        install_text = f"Installing {module_name}..."
        print(install_text)
        progress_bar.labelText = install_text
        slicer.app.processEvents()

        pip_install(module_name)
        install_text = f"{module_name} installed !"
        print(install_text)
        progress_bar.labelText = install_text
        progress_bar.value = math.floor(((i + 1) / len(modules)) * 100)
        slicer.app.processEvents()

    progress_bar.close()


def check_and_install_missing_dependencies() -> None:
    """
    Check if the user has all the required dependencies.
    If not, install them.
    """

    missing_modules = check_missing_module_pip(required_modules)

    if missing_modules:
        with slicer.util.tryWithErrorDisplay(
            "Failed to install dependencies.", waitCursor=True
        ):
            install_missing_module(missing_modules)
