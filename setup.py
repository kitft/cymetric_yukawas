import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

with open("VERSION", "r") as fh:
    VERSION = fh.read()

with open("requirements.txt", "r") as fh:
    REQUIREMENTS = fh.read().splitlines()

setuptools.setup(
    name="cymetric",
    version=VERSION,
    author="Fabian Ruehle, Robin Schneider, Kit FT",
    author_email="cristofero.fraser-taliente@physics.ox.ac.uk",
    description="An open source python library to study CY metrics.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/kitft/cymetric_yukawas",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=REQUIREMENTS,
    package_data={'cymetric': ['wolfram/*.m']},
)
