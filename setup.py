from setuptools import setup, find_packages

setup(
    name='molytica_m',
    version='0.1',
    author='Oliver Midbrink',
    author_email='oliver.midbrink@stud.ki.se',
    description='Molytica M Software',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'requests',
        'spektral',
        'networkx',
        'biopython',
        'torch',
        'torch_geometric',
        'openai',
        'scanpy'
    ],
)
