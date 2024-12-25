from distutils.core import setup

setup(
    name='elsciRL',
    version='0.0.9',
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
        'elsciRL.experiments',
        'elsciRL.instruction_following',
        'elsciRL.interaction_loops',
        'elsciRL.benchmarking_suite',
        ],
    # TODO: Add benchmark to exclusion of wheel
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
        'stable-baselines3'

    ] 
)
