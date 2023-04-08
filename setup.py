import os
import re
from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

pkg_name = "eevalue-tool"
name = "eevalue"
entry_point = "eevalue = eevalue.__main__:main"

here = Path(os.path.dirname(__file__)).resolve()


# get package version
with open(os.path.join(str(here), name, "__init__.py"), encoding="utf-8") as f:
    result = re.search(r'__version__ = ["\']([^"\']+)', f.read())
    if not result:
        raise ValueError(f"Can't find the version in {name}/__init__.py")
    version = result.group(1)


# get the dependencies and installs
with open("requirements.txt", encoding="utf-8") as f:
    requires = [x.strip() for x in f if x.strip()]

# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    readme = f.read()

data_files = []
for pattern in ["**/*", "**/.*", "**/.*/**", "**/.*/.**"]:
    data_files.extend(
        [
            name.replace("eevalue/", "", 1)
            for name in glob("eevalue/config/" + pattern, recursive=True)
        ]
    )

configuration = {
    "name": pkg_name,
    "version": version,
    "python_requires": ">=3.7",
    "description": "A library for estimating the value of energy efficiency as a grid resource",
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
    ],
    "keywords": "power market, power system, simulation, calibration",
    "url": "https://github.com/hebes-io/eevalue",
    "packages": find_packages(),  # include all packages under src
    "entry_points": {"console_scripts": [entry_point]},
    "install_requires": requires,
    "include_package_data": True,
    "package_data": {name: data_files},
}

setup(**configuration)
