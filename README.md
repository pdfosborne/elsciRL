# elsciRL

<a href="https://elsci.org"><img src="https://github.com/pdfosborne/elsciRL-Wiki/blob/main/Resources/images/elsci-whale-logo.png" align="left" height="160" width="160" ></a> 

<div align="center">
  <b>Automate Anything, Automate Everything!</b>
  <br>
  Visit our <a href="https://elsci.org">website</a> to get started, explore our <a href="https://github.com/pdfosborne/elsciRL-Wiki">open-source Wiki</a> to learn more or join our <a href="https://discord.gg/GgaqcrYCxt">Discord server</a> to connect with the community.
  <br>
  <i>In early Alpha Development.</i>
</div>

<div align="center">  
  <br>

  <a href="https://github.com/pdfosborne/elsciRL">![elsciRL GitHub](https://img.shields.io/github/stars/pdfosborne/elsciRL?style=for-the-badge&logo=github&label=elsciRL&link=https%3A%2F%2Fgithub.com%2Fpdfosborne%2FelsciRL)</a>
  <a href="https://github.com/pdfosborne/elsciRL-Wiki">![Wiki GitHub](https://img.shields.io/github/stars/pdfosborne/elsciRL-Wiki?style=for-the-badge&logo=github&label=elsciRL-Wiki&link=https%3A%2F%2Fgithub.com%2Fpdfosborne%2FelsciRL-Wiki)</a>
  <a href="https://discord.gg/GgaqcrYCxt">![Discord](https://img.shields.io/discord/1310579689315893248?style=for-the-badge&logo=discord&label=Discord&link=https%3A%2F%2Fdiscord.com%2Fchannels%2F1184202186469683200%2F1184202186998173878)</a>

</div>

## What is elsciRL?

**elsciRL (pronounced L-SEE)** is an abbreviation of our mission: *"Everything can be automated using Language and Self-Completing Instructions in Reinforcement Learning"*.

To achieve this, **elsciRL** offers a novel framework and infrastructure for accelerating the development of language based Reinforcement Learning solutions.

## Install Guide

1. Install the Python library
```
pip install elsciRL
```
2. Test the install has worked by running the demo experiment in a Python shell or as a script
```python
from elsciRL import DemoExperiment
exp = DemoExperiment()
exp.run()
exp.evaluate()
``` 
*This will run a Reinforcement Learning experiment on two simple problems (OpenAI Gym's FrozenLake and a Sailing Simulation).*

![demo\_gif](https://raw.githubusercontent.com/pdfosborne/elsciRL-Wiki/refs/heads/main/Documentation/I%20-%20Introduction/attachments/elsciRL_demo_short.gif)

*The analyze function will return a combined chart for the experiments you just ran!*

![variance\_comparison\_TRAINING](https://raw.githubusercontent.com/pdfosborne/elsciRL-Wiki/refs/heads/main/Documentation/I%20-%20Introduction/attachments/variance_comparison_TRAINING.png)

## App Interface Setup

[**DEMO VIDEO**](https://www.youtube.com/embed/JbPtl7Sk49Y)

Simply run the following code in a Python script.

```python
from elsciRL import App
App.run()
```

This will give you a localhost link to open in a browser. 

The App will looks similar to the [public demo](https://osbornep.pythonanywhere.com/) except now the algorithm will directly compare your instruction against known environment positions to find the best match. Furthermore, if validated as a correct match, will use your computer resources to train the agent with this factored in.

By default the instruction agent will be run for 1,000 episodes to save time but you can increase or decrease this if you wish to change the training time.

Once you've finished a test you can see the results will show on the App and also save the full output in your local file directory.

---

# Cite

Please use the following to cite this work

```bibtex
  title        = {Improving Real-World Reinforcement Learning by Self Completing Human Instructions on Rule Defined Language},  
  author       = {Philip Osborne},  
  year         = 2024,  
  month        = {August},  
  address      = {Manchester, UK},  
  note         = {Available at \url{https://research.manchester.ac.uk/en/studentTheses/improving-real-world-reinforcement-learning-by-self-completing-hu}},  
  school       = {Department of Computer Science},  
  type         = {PhD thesis}
```

