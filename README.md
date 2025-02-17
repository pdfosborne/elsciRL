# elsciRL
## Language Guidance Framework for Reinforcement Learning

<a href="https://elsci.org"><img src="https://raw.githubusercontent.com/pdfosborne/elsciRL-Wiki/refs/heads/main/Resources/images/elsciRL_logo_owl_cropped.png" align="left" height="250" width="250" ></a>

<div align="center">
  <br>
  <b>Open-source Python Software for Academic and Industry Applications</b>

  Visit our <a href="https://elsci.org">website</a> to get started, explore our <a href="https://github.com/pdfosborne/elsciRL-Wiki">open-source Wiki</a> to learn more or join our <a href="https://discord.gg/GgaqcrYCxt">Discord server</a> to connect with the community.
  <br>
  <i>In pre-alpha development.</i>
  <p> </p>
</div>

<div align="center">  

  <a href="https://github.com/pdfosborne/elsciRL">![elsciRL GitHub](https://img.shields.io/github/stars/pdfosborne/elsciRL?style=for-the-badge&logo=github&label=elsciRL&link=https%3A%2F%2Fgithub.com%2Fpdfosborne%2FelsciRL)</a>
  <a href="https://github.com/pdfosborne/elsciRL-Wiki">![Wiki GitHub](https://img.shields.io/github/stars/pdfosborne/elsciRL-Wiki?style=for-the-badge&logo=github&label=elsciRL-Wiki&link=https%3A%2F%2Fgithub.com%2Fpdfosborne%2FelsciRL-Wiki)</a>
  <a href="https://discord.gg/GgaqcrYCxt">![Discord](https://img.shields.io/discord/1310579689315893248?style=for-the-badge&logo=discord&label=Discord&link=https%3A%2F%2Fdiscord.com%2Fchannels%2F1184202186469683200%2F1184202186998173878)</a> 
  
  <b>Quicklinks:</b> [Homepage](https://elsci.org) | [FAQs](https://elsci.org/FAQs) | [New Developers](https://elsci.org/New+Developers) | [Contributing Guide](https://elsci.org/Become+a+Contributor) | [App Interface Guide](https://elsci.org/App+Interface+Guide)
  <br>
  <br>
</div>
<div align="left">


## What is elsciRL?

**elsciRL (pronounced L-SEE)**** offers a general purpose Python library for accelerating the development of language based Reinforcement Learning (RL) solutions.

Our novel solution is a patent pending framework that allows end users to give instructions to Reinforcement Learning agents without direct supervision. In other words, prior methods required any objectives to be hard coded as rules or shown by demonstration (e.g. if key in inventory then objective reached). 

This has also been known as AI agents but we notably do not require the problem to already contain language.

### Features
1. **Accelerates Research** by seperating Reinforcement Learning development into distinct components and providing an [Open-Source Wiki](https://github.com/pdfosborne/elsciRL-Wiki) and [Discord Server](https://discord.gg/GgaqcrYCxt) to share knowledge
2. **Improve Reproducability** by designing generally applicable alogorithms & applications including configurations to re-create prior experiments
3. **Extract Domain Expert Knowledge** by using our App Interface to let non-technical users provide instructions
4. **Enchance New Applications** by making it easier to setup new problems and reduce the amount of data required testing

<div width="75%" align="center">
	<img src="https://github.com/pdfosborne/elsciRL-Wiki/blob/main/Resources/images/Agent-Performance-2.gif?raw=true" />
</div>

### What is Reinforcement Learning?

Reinforcement Learning is an Artificial Intelligence methodology that teaches machines how to make decisions and perform actions to achieve a goal.

It's based on the idea that machines can learn from their experiences to automate a task without being told exactly what to do, similar to how humans learn through trial and error.

See the [FAQs](https://elsci.org/FAQs) for more information.

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

## Run the App Interface

To run the App, run the following code in a Python script.

```python
from elsciRL import App
App.run()
```

This will give you a localhost link to open the App in a browser. 


*Click the image to watch the demo video*

[![YouTube](https://github.com/pdfosborne/elsciRL-Wiki/blob/main/Resources/images/elsciRL-WebApp-Demo.png?raw=true)](https://www.youtube.com/watch?v=JbPtl7Sk49Y)


The App will looks similar to the [public demo](https://osbornep.pythonanywhere.com/) except now the algorithm will directly compare your instruction against known environment positions to find the best match. Furthermore, if validated as a correct match, will use your computer resources to train the agent with this factored in.

By default the instruction agent will be run for 1,000 episodes to save time but you can increase or decrease this if you wish to change the training time.

Once you've finished a test you can see the results will show on the App and also save the full output in your local file directory.

---

# Cite

Please use the following to cite this work

```bibtex
@phdthesis{Osborne2024,
  title        = {Improving Real-World Reinforcement Learning by Self Completing Human Instructions on Rule Defined Language},  
  author       = {Philip Osborne},  
  year         = 2024,  
  month        = {August},  
  address      = {Manchester, UK},  
  note         = {Available at \url{https://research.manchester.ac.uk/en/studentTheses/improving-real-world-reinforcement-learning-by-self-completing-hu}},  
  school       = {The University of Manchester},  
  type         = {PhD thesis}
}
```

