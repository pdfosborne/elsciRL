# Removed from init for now as it pre-loads data that is not needed for the package to run
#from .examples.experiment import DemoExperiment
from .examples.WebApp.app import app as Demo
from .GUI.app import app as App

from .experiments.standard import Experiment as STANDARD_RL
from .instruction_following.supervised_instruction_following import SupervisedExperiment as SUPERVISED_RL_HIERARCHY
from .instruction_following.unsupervised_instruction_following import UnsupervisedSearch as UNSUPERVISED_RL_HIERARCHY
from .instruction_following.elsciRL_instruction_search import elsciRLSearch as elsciRL_SEARCH
from .instruction_following.elsciRL_instruction_following import elsciRLOptimize as elsciRL_OPTIMIZE

from .analysis.combined_variance_visual import combined_variance_analysis_graph as COMBINED_VARIANCE_ANALYSIS_GRAPH
