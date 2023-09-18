from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="mfit",
    version='1.0.0',
    description="MMRT Fitting Tool",
    author="Carlin Hamill",
    author_email="carlinhamill.mmrt@gmail.com",
    url="",
    license="GPLv3",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    keywords=["Mfit", "MMRT"],
    include_package_data=True,
    package_data={'mfit': ['mfit1/Help.pdf', 'mfit2/Help.pdf']},
    packages=[
        "mfit",
        "mfit.mfit1",
        "mfit.mfit1.ui",
        "mfit.mfit2",
        "mfit.mfit2.ui"
    ],
    python_requires=">=3.8",
    install_requires=[
        "scipy>=1.10.1",
        "PyQt6>=6.5.2",
        "numpy>=1.24.4, <1.25",
        "openpyxl>=3.1.2",
        "pandas>=2.0.0",
        "lmfit>=1.2.0",
        "numba>=0.57.1",
        "matplotlib>=3.7.1"
    ],
    entry_points={
        "console_scripts": [
            "mfit1=mfit.__main__:mfit1",
            "mfit2=mfit.__main__:mfit2"
        ],
    },
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)