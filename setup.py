from setuptools import setup, find_packages


# Also install sudo apt-get install openbabel

setup(
    name='molytica_m',
    version='0.1',
    author='Oliver Midbrink',
    author_email='oliver.midbrink@stud.ki.se',
    description='Molytica M Software',
    packages=find_packages(),
    install_requires=[ # a
        'numpy',
        'pandas',
        'requests',
        'spektral',
        'networkx',
        'biopython',
        'torch',
        'torch_geometric',
        'openai',
        'scanpy',
        'requests',
        'beautifulsoup4',
    ],
)
