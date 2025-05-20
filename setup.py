# from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='elsciRL',
    version='0.3.1',
    packages=[
        'elsciRL', 
        'elsciRL.adapters',
        'elsciRL.agents',
        'elsciRL.agents.stable_baselines',
        'elsciRL.analysis',
        'elsciRL.encoders', 
        'elsciRL.environment_setup',
        'elsciRL.evaluation',
        'elsciRL.examples',
        'elsciRL.examples.adapters',
        'elsciRL.examples.environments',
        'elsciRL.examples.local_configs',
        'elsciRL.GUI',
        'elsciRL.GUI.static',
        'elsciRL.GUI.templates',
        'elsciRL.GUI.LLM',
        'elsciRL.experiments',
        'elsciRL.instruction_following',
        'elsciRL.interaction_loops',
        'elsciRL.application_suite',
        ],
    package_data={
        'elsciRL.GUI.templates': ['index.html'],
        'elsciRL.GUI.static': ['styles.css', 'app_setup.md'],
    },
    include_package_data=True,
    url='https://github.com/pdfosborne/elsciRL',
    license='Apache-2.0 license',
    author='Philip Osborne',
    author_email='pdfosborne@gmail.com',
    description='Applying the elsciRL architecture to Reinforcement Learning problems.',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy>=1.10.1',
        'torch',
        'tqdm',
        'httpimport',
        'sentence-transformers',
        'gymnasium',
        'stable-baselines3',
        'flask'
    ] 
)
