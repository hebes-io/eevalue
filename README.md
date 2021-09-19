<img src="https://github.com/hebes-io/eevalue/blob/main/logo.png" width="400" height="200">
<br/><br/>

## The `eevalue` tool for estimating the value of energy efficiency in conditions of missing capacity

The `eevalue` Python package accompanies the deliverable <b>D4.2 The drivers of the value of energy efficiency as an energy resource</b> of the H2020 project [SENSEI Smart Energy Services to Improve the Energy Efficiency of the European Building Stock](https://senseih2020.eu/). 

The energy efficiency services sector is lagging behind renewable energy generation in terms of demand for investments, as well as business models for aggregating projects and attracting investment capital from institutional investors. The term “institutional investors” is an umbrella term covering pension funds, insurance companies, banking institutions and other investment funds. While mature business models for financing and implementing energy efficiency models at the level of individual buildings already exist, scaling energy efficiency up to project portfolio level still faces challenges; there is still a need for replicable approaches to create large-scale pipelines of projects that can be aggregated to appeal to third-party financiers.

The main argument of SENSEI is twofold:

1. The roll out of Pay-for-Performance (P4P) pilots for energy efficiency projects can lay the groundwork for the innovations in contract design and performance evaluation that are necessary for creating such large-scale pipelines of projects. In particular, P4P pilots constitute an effective use of public finance to discover best practices for the aggregation of a large number of energy efficiency projects into portfolios, and they can act as a workbench for developing financing tools and risk allocation mechanisms in order to increase demand for energy efficiency projects. 


2. P4P pilots can stem from a policy decision to align energy efficiency support schemes with the foreseen needs of the power grid. The value of such an alignment comes from the interactions between energy efficiency improvements in buildings and the power system’s need for peak capacity and/or demand flexibility. An energy efficiency measure (EEM) may reduce demand during the hours when the probability of load loss is high. Simultaneously, there are times when increased demand may be beneficial (e.g., during periods of renewable energy over-generation and curtailment). If energy efficiency reduces demand in those hours, then the system need for flexibility will increase.  

SENSEI acknowledges that power system operators cannot directly compensate the energy efficiency measures and cannot directly monitor their performance – both functions are out of their scope and responsibilities. Furthermore, and since energy efficiency is not seen or optimized by the energy or capacity market, it cannot receive a qualifying capacity value for resource adequacy purposes. Instead, it must be compensated through programs that identify and support the measures that are most effective in offsetting the need for new generating plants or transmission upgrades. As a result, the coordination between the needs of the power system and the incentives for energy efficiency improvements must take place during the (medium-term) planning for resource adequacy in the power system. 

The first step towards defining a P4P scheme that links energy efficiency in buildings with the power system’s state is to define what constitutes a load modifier resource. Load-modifiers are those resources or programs not seen or optimized by the power or capacity market, that persistently modify the power system’s load shape in ways that harmonize with the system operator’s goals. An effective load modifying resource helps create a flatter system load profile, attenuating high power peaks and valleys and reducing extreme upward and downward ramps.

In order to link energy efficiency with the challenges of the power system’s operation, this deliverable makes the working assumption energy efficiency improvements are valorised through a Pay for Load Shape (P4LS) program, as proposed by the California Public Utilities Commission’s Working Group on Load Shift. A P4LS program: 

* Operates outside of the power and capacity market;

* Is based on target load shapes that change gradually according to the evolving conditions of the grid;

* Is aligned with the P4P concept, since an energy efficiency project is compensated according to its actual impact and the changes in energy consumption that occur because of it; 

* Incentivises decisions that have a persistent effect on the daily and seasonal profile of electricity demand, such as equipment upgrades, installation of control technologies and building envelope improvements;

* Makes it possible to measure the performance of an energy efficiency project in a way that is similar to the way demand response is measured: the minimum amount of “work” required to transform the baseline consumption profile to the requested one.

The `eevalue` python package includes all necessary functionality to automatically build and validate a unit commitment model to derive target load shapes for a hypothetical P4LS program. The utilized unit commitment model is largely based on the Linear Programming (LP) formulation of the [Dispa-SET model](https://github.com/energy-modelling-toolkit/Dispa-SET/) that is developed within the Joint Research Centre of the European Commission.
<br>

<img align="left" width="500" src="https://github.com/hebes-io/eensight/blob/master/EC_support.png">
