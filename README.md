<!-- # elsciRL -->
<!-- ## Integrating Language Solutions into Reinforcement Learning -->
<div align="center">
  <h1>Improve Any Reinforcement Learning Application with Language</h1>
</div>

<a href="https://elsci.org"><img src="https://raw.githubusercontent.com/pdfosborne/elsciRL-Wiki/refs/heads/main/Resources/images/elsciRL_julylogo_black_outline.png" align="left" height="200" width="200" ></a>

<div align="center">
  <b>Open-source Python Software for Academic and Industry Applications</b>

  Visit our <a href="https://elsci.org">website</a> to get started, explore our <a href="https://github.com/pdfosborne/elsciRL-Wiki">open-source Wiki</a> to learn more or join our <a href="https://discord.gg/GgaqcrYCxt">Discord server</a> to connect with the community.
  <br>
  <i>In pre-alpha development.</i>
  <p> </p>
</div>

<div align="center">  

  <a href="https://github.com/pdfosborne/elsciRL">![elsciRL GitHub](https://img.shields.io/github/stars/pdfosborne/elsciRL?style=for-the-badge&logo=github&label=elsciRL&link=https%3A%2F%2Fgithub.com%2Fpdfosborne%2FelsciRL)</a>
  <a href="https://github.com/pdfosborne/elsciRL-Wiki">![Wiki GitHub](https://img.shields.io/github/stars/pdfosborne/elsciRL-Wiki?style=for-the-badge&logo=github&label=elsciRL-Docs&link=https%3A%2F%2Fgithub.com%2Fpdfosborne%2FelsciRL-Wiki)</a>
  <a href="https://discord.gg/GgaqcrYCxt">![Discord](https://img.shields.io/discord/1310579689315893248?style=for-the-badge&logo=discord&label=Discord&link=https%3A%2F%2Fdiscord.com%2Fchannels%2F1184202186469683200%2F1184202186998173878)</a> 

  <b>Quicklinks:</b> [Homepage](https://elsci.org) | [FAQs](https://elsci.org/FAQs) | [New Developers](https://elsci.org/New+Developers) | [Contributing Guide](https://elsci.org/Become+a+Contributor) | [App Interface Guide](https://elsci.org/App+Interface+Guide)
  <br>
  <br>
</div>

---
### Announcements
[July 2025] elsciRL update 0.3.6 LLM methods now available and re-usability improvements to the interface.

[July 2025] **New publication** [*elsciRL: Integrating Language Solutions into Reinforcement Learning Problem Settings*](https://arxiv.org/abs/2507.08705)

---
<div align="center">
  
  <a href="https://www.youtube.com/watch?v=JbPtl7Sk49Y">![GUI_Preview_GIF](https://raw.githubusercontent.com/pdfosborne/elsciRL-Wiki/refs/heads/main/Resources/images/elsciRL_GUI_GIF_2.gif)</a>
  <a href="https://www.youtube.com/@DrPhilipOsborne">![YouTube](https://img.shields.io/youtube/channel/views/UCJo8IlRyjvxmHdyt_begm8Q?style=for-the-badge&logo=youtube&label=YouTube&link=https%3A%2F%2Fwww.youtube.com%2F%40DrPhilipOsborne)</a>
</div>
<div align="left">


<div align="left" style="font-size:150%">
	<p><b>Features</b></p>	
</div>

1. **Enhance any Reinforcement Learning application with language.** 
2. Our **Graphical User Interface (GUI)** makes it easy to apply new algorithms and provide instructions.
3. **Develop applications faster** with fewer problem specific requirements.
4. **Accelerate your research** with our [Open-Source Wiki](https://github.com/pdfosborne/elsciRL-Wiki) and [Discord Server](https://discord.gg/GgaqcrYCxt) to share knowledge.
5. **Enables  reproducibility of your work** including publication of the configurations used so others can re-create the experiments.
6. **Collect user input data with ease** by using our GUI to let non-technical users provide instructions that guide the agent.

<div align="center">
	<img src="https://raw.githubusercontent.com/pdfosborne/elsciRL-Wiki/refs/heads/main/Resources/images/elsciRL_comparison_flow.png" />
</div>

---
## Install Guide

### Quick Install

It is suggested to use a [Python environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#). 

Then, install the Python library from the PyPi package library:

```bash
pip install elsciRL
```

### Manual Install
Alternatively, you can clone this repository directly and install it manually.

```bash
git clone https://github.com/pdfosborne/elsciRL.git
cd elsciRL
pip install .
```

### Developer Install
If you wish wish to edit the software you can do this my installing it as an editable package.

```bash
git clone https://github.com/pdfosborne/elsciRL.git
cd elsciRL
pip install -e .
```

<!-- ## Quick Demo

To check the install has worked, you can run a quick CLI demo from a selection of applications:

```python
from elsciRL import Demo
test = Demo()
test.run()
```

This will run a tabular Q learning agent on your selected problem and save results to:

> '*CURRENT_DIRECTORY*/elsciRL-EXAMPLE-output/...'

A help function is included in demo: *test.help()* -->


## GUI App

To run the App, run the following code in a Python script.

```python
from elsciRL import App
App.run()
```

This will give you a localhost link to open the App in a browser. 

[![YouTube](https://github.com/pdfosborne/elsciRL-Wiki/blob/main/Resources/images/elsciRL-WebApp-Demo-YTlogo.png?raw=true)](https://www.youtube.com/watch?v=JbPtl7Sk49Y)
<div width="75%" align="center">
	<p><i>The Home tab provides an guide of how to use the GUI. Click the image to watch the demo video.</i></p>
</div>

## Applications
Each application page provides a summary, configuration options, available adapters, and citation information.


| Application                | Year | Repo Name                | Published by |
| :------------------------: | :---------------------: | :---------------------: | :----------: | 
| [Maze Navigation](https://github.com/pdfosborne/elsciRL-App-Maze) | 2025 | elsciRL-Maze            | pdfosborne | 
| [Sailing Simulation](https://github.com/pdfosborne/elsciRL-App-Sailing) | 2024 | elsciRL-Sailing         | pdfosborne |  
| [Chess Simulation](https://github.com/pdfosborne/elsciRL-App-Chess) | 2024 | elsciRL-Chess           | pdfosborne |                        |
| [GridWorld Classroom](https://github.com/pdfosborne/elsciRL-App-Classroom)  | 2024 | elsciRL-Classroom       | pdfosborne | 
| [Gym-FrozenLake](https://github.com/pdfosborne/elsciRL-App-GymFrozenLake)    | 2024 | elsciRL-GymFrozenLake   | pdfosborne | 
| [TextWorldExpress](https://github.com/pdfosborne/elsciRL-App-TextWorldExpress)  | 2024 | elsciRL-TextWorldExpress| pdfosborne |        

---

## What is elsciRL?

**elsciRL (pronounced L-SEE)** offers a general purpose Python library for accelerating the development of language based Reinforcement Learning (RL) approaches.

Our novel solution is a two-phased approach.

First, we use adapters to introduce language with reduced setup requirements. For example, this may be in the form of a set of one-to-one mapping rules to more advanced prediction methods to transform environment states to language. 

Second, now that language exists in the problem space, we can then apply our unsupervised instruction following approach. Prior approaches required some form of human labelling efforts to complete instructions which can be costly to obtain. 

These combined define our **Explicit Language for Self-Completing Instruction** methodology, i.e. **elsci**.

Our solution allows end users to give instructions to Reinforcement Learning agents without direct supervision where prior methods required any objectives to be hard coded as rules or shown by demonstration (e.g. if key in inventory then objective reached). 

Our work fits within the scope of AI agents but we notably do not require the problem to already contain language which is normally required for applying LLMs.

<div width="75%" align="center">
	<img src="https://raw.githubusercontent.com/pdfosborne/elsciRL-Wiki/refs/heads/main/Resources/images/elsciRL_LLM_Overview-v2.png" />
	<p><i>Overview of the elsciRL library, <b style='color:red;'>red blocks</b> highlight the Adapter and Self-completing Instruction Following methodologies, <b style='color:blue;'>blue blocks</b> highlight our most recent developments with LLMs.</i></p>
</div>

### What is Reinforcement Learning?

Reinforcement Learning is an Artificial Intelligence methodology that teaches machines how to make decisions and perform actions to achieve a goal.

It's based on the idea that machines can learn from their experiences to automate a task without being told exactly what to do, similar to how humans learn through trial and error.

See the [FAQs](https://elsci.org/FAQs) for more information.


---

### Cite

Please use the following to cite this work

```bibtex
@phdthesis{OsborneThesis2024,
  title        = {Improving Real-World Reinforcement Learning by Self Completing Human Instructions on Rule Defined Language},  
  author       = {Philip Osborne},  
  year         = 2024,  
  month        = {August},  
  address      = {Manchester, UK},  
  note         = {Available at \url{https://research.manchester.ac.uk/en/studentTheses/improving-real-world-reinforcement-learning-by-self-completing-hu}},  
  school       = {The University of Manchester},  
  type         = {PhD thesis}
}

@misc{elsciRL2025,
      title={elsciRL: Integrating Language Solutions into Reinforcement Learning Problem Settings}, 
      author={Philip Osborne and Danilo S. Carvalho and André Freitas},
      year={2025},
      eprint={2507.08705},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.08705}, 
}
```









