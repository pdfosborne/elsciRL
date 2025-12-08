# Try to import modules, handle missing dependencies gracefully
try:
    from .examples.DemoExperiment import DemoExperiment as Demo
except ImportError as e:
    print(f"Warning: Could not import DemoExperiment: {e}")
    Demo = None

try:
    from .GUI.app import app as App
except ImportError as e:
    print(f"Warning: Could not import GUI app: {e}")
    App = None

try:
    from .GUI.prerender import Prerender as get_prerender_data
except ImportError as e:
    print(f"Warning: Could not import Prerender: {e}")
    get_prerender_data = None

try:
    from .experiments.standard import Experiment as STANDARD_RL
except ImportError as e:
    print(f"Warning: Could not import STANDARD_RL: {e}")
    STANDARD_RL = None

try:
    from .instruction_following.elsciRL_instruction_search import elsciRLSearch as elsciRL_SEARCH
except ImportError as e:
    print(f"Warning: Could not import elsciRL_SEARCH: {e}")
    elsciRL_SEARCH = None

try:
    from .instruction_following.elsciRL_instruction_following import elsciRLOptimize as elsciRL_OPTIMIZE
except ImportError as e:
    print(f"Warning: Could not import elsciRL_OPTIMIZE: {e}")
    elsciRL_OPTIMIZE = None

try:
    from .analysis.combined_variance_visual import combined_variance_analysis_graph as COMBINED_VARIANCE_ANALYSIS_GRAPH
except ImportError as e:
    print(f"Warning: Could not import COMBINED_VARIANCE_ANALYSIS_GRAPH: {e}")
    COMBINED_VARIANCE_ANALYSIS_GRAPH = None
