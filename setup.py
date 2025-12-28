"""
Setup script for CrystalMOE-LLM
ALIGNN MoE + ZeroMAT Integration
"""

from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements(filename='requirements.txt'):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="crystalmoe-llm",
    version="0.1.0",
    description="ALIGNN Mixture-of-Experts + ZeroMAT for Crystal Materials",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Crystal MoE Team",
    python_requires=">=3.8",
    packages=find_packages(exclude=['tests', 'scripts', 'configs', 'checkpoints', 'data']),

    install_requires=[
        # Core deep learning
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0",

        # Graph/Crystal libraries
        "dgl>=1.1.0",
        "jarvis-tools>=2022.9.16",
        "alignn>=2022.9.15",
        "pymatgen>=2023.5.10",

        # 3D Molecule (existing ZeroMAT)
        "torch-geometric",

        # Utilities
        "pyyaml",
        "pandas",
        "numpy",
        "tqdm",
        "scikit-learn",
    ],

    extras_require={
        'dev': [
            'pytest',
            'black',
            'flake8',
            'ipython',
            'jupyter',
        ],
    },

    entry_points={
        'console_scripts': [
            'crystalmoe-train=scripts.train_moe_stage2:main',
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],

    keywords="materials science, machine learning, mixture-of-experts, crystal structure, property prediction",
)
