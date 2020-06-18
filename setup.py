from setuptools import find_packages, setup

setup(
    name='gaitkeeper',
    packages=find_packages(),
    version='0.1.0',
    description='Gait-based authentication project.',
    author='Fred Fu',
    license='MIT',
    # Packages should be installed with conda (because torch requires CUDA), but
    # this will check that the important ones exist at least.
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-learn',
        'torch',
        'torchvision',
        'flask',
    ]
)
