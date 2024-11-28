from distutils.core import setup

setup(
    name='elsciRL',
    version='1.3.26',
    packages=[
        'elsciRL', 
        'elsciRL.agents',
        'elsciRL.analysis',
        'elsciRL.benchmarking_suite',
        'elsciRL.encoders', 
        'elsciRL.environment_setup',
        'elsciRL.evaluation',
        'elsciRL.examples',
        'elsciRL.experiments',
        'elsciRL.instruction_following',
        'elsciRL.interaction_loops',
        ],
    # TODO: Add benchmark to exclusion of wheel
    url='https://github.com/pdfosborne/elsciRL',
    license='GNU Public License v3',
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
        'sentence-transformers',
        'tqdm',
        'httpimport'
    ] 
)
