
<a href="url"><img src="https://github.com/pdfosborne/elsciRL-Wiki/blob/main/Resources/images/elsciRL_logo.png" align="left" height="48" width="48" ></a>

# elsciRL (pronounced L-SEE)

*Every problem can be automated with Language and Self-Completing Instructions.*

[![GitHub watchers](https://img.shields.io/github/watchers/pdfosborne/elsciRL-Wiki?style=for-the-badge&logo=github&label=elsciRL-Wiki&link=https%3A%2F%2Fgithub.com%2Fpdfosborne%2FelsciRL-Wiki)](https://github.com/pdfosborne/elsciRL-Wiki)  [![Discord](https://img.shields.io/discord/1310579689315893248?style=for-the-badge&logo=discord&label=Discord&link=https%3A%2F%2Fdiscord.com%2Fchannels%2F1184202186469683200%2F1184202186998173878)](https://discord.gg/A2dRVrhB)


This is the homepage of applying the elsciRL system to any Reinforcement Learning problem. 

Complete Documentation can be found here: [elsciRL-Wiki](https://github.com/pdfosborne/elsciRL-Wiki).

To get started, see [Getting Started](https://github.com/pdfosborne/elsciRL-Wiki/blob/main/elsciRL%20Core/I%20-%20Introduction/1%20-%20Getting%20Started.md).

The elsciRL approach is a generally applicable instruction following method whereby a two-layer hierarchy is formed: 1) a high level instruction plan and 2) the low level environment interaction. Uniquely, this work does not assume that instructions (or sub-goals) need to be supervised. Instead, we assume the environment contains some language such that this can be completed unsupervised. Furthermore, the unsupervised completion of each instruction is presented back to the user and their feedback strengthens the quality of the matching between observed environment positions and expected outcomes.

Instructions help greatly in mitigating the issue of long-term objectives never being reached and enable transfer of knowledge through the re-use of sub-instructions to new tasks. 

To make this work possible, we built this software solution such that it could enable the application of this work to any Reinforcement Learning problem. Unlike other Reinforcement Learning packages that are only designed to enable the importing a pre-built agent we go much further. 

First, we standardize the interaction loop setup such that setting up new problems is significantly easier and faster. 

Second, instead of simplifying importing pre-built agents into a custom system we reverse this process so the interaction loop can be imported into far more complex hierarchical solutions without needing to be purpose built to each problems. 

Lastly, analysis formatting and structure is generated such that the individual user only needs to interpret them to adjust parameters accordingly.

Provided a user can setup their problem using the template structure provided they can then leverage the most advanced Reinforcement Learning approaches with a simple parameter input. This also ensures the system is future-proof as new agents or encoders will be added as modules in later updates. 

