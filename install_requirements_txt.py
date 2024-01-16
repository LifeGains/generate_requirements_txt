import subprocess
import pkg_resources
import os

def get_installed_packages():
    """
    Returns a set of installed package names.
    """
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    return installed_packages

def install_requirements(file_path):
    """
    Install packages from a requirements.txt file if they are not already installed.

    Parameters:
    file_path (str): Path to the requirements.txt file.
    """
    installed_packages = get_installed_packages()

    with open(file_path, 'r') as file:
        for line in file:
            package = line.strip()
            # Assume package name is before '==' and skip if no package name is present
            package_name = package.split('==')[0] if '==' in package else package
            if package_name and package_name not in installed_packages:
                subprocess.call(['pip', 'install', package])

# Get the current working directory
current_directory = os.getcwd()

# Set the path to the requirements.txt file in the current directory
requirements_file_path = os.path.join(current_directory, 'requirements.txt')

install_requirements(requirements_file_path)
