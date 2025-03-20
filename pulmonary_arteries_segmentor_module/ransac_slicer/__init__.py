from .dependencies_checker import check_and_install_missing_dependencies
import slicer

restart_needed = check_and_install_missing_dependencies()
if restart_needed:
    slicer.app.restart()
