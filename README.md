# elsciRL
## Improve Reinforcement Learning with Language AI

<a href="https://elsci.org"><img src="https://raw.githubusercontent.com/pdfosborne/elsciRL-Wiki/refs/heads/main/Resources/images/elsciRL_logo_stag_transparent_cropped_fix.png" align="left" height="175" width="160" ></a> 

<div align="left">
  <br>
  Visit our <a href="https://elsci.org">website</a> to get started, explore our <a href="https://github.com/pdfosborne/elsciRL-Wiki">open-source Wiki</a> to learn more or join our <a href="https://discord.gg/GgaqcrYCxt">Discord server</a> to connect with the community.
  <br>
  <i>In pre-alpha development.</i>
</div>

<div align="left">  
  <br>

  <a href="https://github.com/pdfosborne/elsciRL">![elsciRL GitHub](https://img.shields.io/github/stars/pdfosborne/elsciRL?style=for-the-badge&logo=github&label=elsciRL&link=https%3A%2F%2Fgithub.com%2Fpdfosborne%2FelsciRL)</a>
  <a href="https://github.com/pdfosborne/elsciRL-Wiki">![Wiki GitHub](https://img.shields.io/github/stars/pdfosborne/elsciRL-Wiki?style=for-the-badge&logo=github&label=elsciRL-Wiki&link=https%3A%2F%2Fgithub.com%2Fpdfosborne%2FelsciRL-Wiki)</a>
  <a href="https://discord.gg/GgaqcrYCxt">![Discord](https://img.shields.io/discord/1310579689315893248?style=for-the-badge&logo=discord&label=Discord&link=https%3A%2F%2Fdiscord.com%2Fchannels%2F1184202186469683200%2F1184202186998173878)</a> <a href="https://github.com/pdfosborne/elsciRL">![elsciRL GitHub](https://img.shields.io/github/sponsors/pdfosborne?style=for-the-badge&logo=githubsponsors&logoColor=%23EA4AAA&labelColor=%23FFFFFF&color=%23EA4AAA)</a>  

</div>

## What is elsciRL?

**elsciRL (pronounced L-SEE)** is an abbreviation of our mission: *"Everything can be automated using Language and Self-Completing Instructions in Reinforcement Learning"*.

To achieve this, **elsciRL** offers a novel framework and infrastructure for accelerating the development of language based Reinforcement Learning solutions.

## Install Guide

Install the Python library from the PyPi packagel library
```
pip install elsciRL
```

Alternatively, you can clone this repository directly and install it manually.
```
git clone https://github.com/pdfosborne/elsciRL.git
cd elsciRL
pip install .
```

If you wish wish to edit the software you can do this my installing it as an editable package.
```
git clone https://github.com/pdfosborne/elsciRL.git
cd elsciRL
pip install -e .
```

## Run the App Interface

Simply run the following code in a Python script.

```python
from elsciRL import App
App.run()
```

[![YouTube](http://i.ytimg.com/vi/JbPtl7Sk49Y/hqdefault.jpg)](https://www.youtube.com/watch?v=JbPtl7Sk49Y)

<!-- ![WebApp_example_image_1](https://raw.githubusercontent.com/pdfosborne/elsciRL-Wiki/refs/heads/main/Resources/images/WebApp-example-1.png) -->

<!-- <iframe width="560" height="315" src="https://www.youtube.com/embed/JbPtl7Sk49Y?si=vAEplNNdmXcEDAx-&amp;controls=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe> -->


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

