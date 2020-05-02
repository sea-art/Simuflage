<p align="center">
    <a href="https://www.uva.nl/en">
        <img width="500" src="https://i.imgur.com/HPWb5UX.png"/>
    </a>
</p> 

<br/>
<p align="center">
    <a href="http://admorph.eu/">
        <img width="400" src="https://i.imgur.com/LJgU1Pf.png"/>
    </a>
</p>  

[![Build Status](https://travis-ci.com/sea-art/Simuflage.svg?token=N3rb3wFxBrspLC9Ysuz7&branch=master)](https://travis-ci.com/github/sea-art/DSE_simulator)
[![codecov](https://codecov.io/gh/sea-art/Simuflage/branch/master/graph/badge.svg?token=DJOIKL65KT)](https://codecov.io/gh/sea-art/Simuflage)
[![Repo Size](https://github-size-badge.herokuapp.com/sea-art/Simuflage.svg)](https://github.com/sea-art/Simuflage)
[![Python 3+7 ready](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/)
[![Licence](https://img.shields.io/badge/license-GPL--v3.0-blue.svg)](LICENSE)
## Summary
Simuflage is a high-level embedded system designer and simulator that can be utilized to evaluate design points regarding
the aspects of power output, thermals, power-efficiency and time to failure (TTF).

#### Goals
- High-level embedded system designer
- Fast embedded system simulator
- Support for adaptivity policies and heterogeneous systems
- Easy utilization by design space exploration strategies
- Flexible, adaptable and maintainable

## About
This repository complements a thesis project of the [MSc Software Engineering](http://www.software-engineering-amsterdam.nl) 
master at the University of Amsterdam. This project falls into the scope of [ADMORPH](http://admorph.eu/), an international 
project funded by the [EU Horizon 2020](https://ec.europa.eu/programmes/horizon2020/en) programme to make various types of complex systems more resistant to defects and more secure.



## Running the simulator
```bash
git clone git@github.com:sea-art/Simuflage.git
cd Simuflage
make setup
make run
```

## Libraries used
- [Numpy](https://numpy.org/) - efficient multi-dimensional container of generic data 
- [Scipy](https://scipy.org/) - library used for scientific and technical computing

## Quickstart

### Creation of components
Manual creation of components can be done as follows
```python
c1 = Component(100, (1, 1))
c2 = Component(120, (0, 1))
```
which will create a component with a power capacity of 100 on xy-coordinates (1, 1) 
and another component with a power capacity of 120 at xy-coordinates (0, 1)

The components have to be bundled together in a list in order to create a design point:
```python
components = [c1, c2]
```

### Creation of applications
Manual creation of applications can be done as:
```python
a1 = Application(50)
a2 = Application(40)
```
Which will create two applications, one requires 50 computing power and the other 40.

Similar to the components, all applications have to be bundled together as a list:
```python
applications = [a1, a2]
```

### Mapping applications to components
Applications have to be mapped to components, and can be done as follows:
```python
app_map = [(c1, a1), (c2, a2)]
```
Which maps the application of 50 to component 1, and the application of 40 to component 2.

**NOTE: An application can only be mapped to a single component, but components can run multiple applications**

### Creating the designpoint
In order to create a design point the component, the applications and the mapping of 
applications to components are required. Once these are created, a design point is initialized as:
```python
dp = Designpoint(components, applications, app_map)
```

### Creating and running the simulator
Once a design point is created, the simulator can be used in order to calculate the Time To Failure (TTF) 
of that design point. All that the simulator requires is a designpoint, which can be initialized as:
```python
sim = simulator.Simulator(dp)
sim.run(until_failure=True)
```
When running the simulator it is either possible to run a fixed amount of iterations, 
or to keep running the simulator until a failure has occurred.

## Code structure
```
├───src
│   ├───design
│   └───simulation
│       └───elements
└───tests
    ├───design
    └───simulation
```
All the code about *design points* are positioned in the ```src/design/``` folder.<br>
 
All the code regarding the *simulation* are placed in the ```src/simulation/``` folder.<br>
The ```simulator.py``` can be seen as the *main* function for the simulator. All other files 
(i.e. ```agings.py```, ```components.py``` and ```thermals.py```) provide an ```iterate()``` function
based on some parameters, which the simulator will call each simulation iteration.

This section explains the structure of this project in more detail aiming to provide information
about how to contribute/alter the simulator.


### Definitions
- **Simulation elements** - all the files located at ```src/simulation/elements/``` that desribe the
simulators behaviour. When changing the behavior of the simulator, files in these folders should be added or changed. The ```integrator```
file is used to make sure the elements integrate and work with each other.
- **Integrator** - since the aim of this simulator is to be modular and easily be changable, the simulator behavior
is expected to change frequently. The integrator file (located at ```src/simulation/integrator.py```) is used to specify
how the simulator should behave each timestep based on the ```simulator elements```.

### Simulation
#### Adding simulation functionality
All elements of the simulator are currently:
- ```agings.py```
- ```components.py```
- ```thermals.py```

When adding elements to the simulator, it has to extend the abstract class ```simulator_element.py```.

### Integrator
The integrator, located at ```src/simulation/integrator.py``` is the file to edit the simulator functionality
upon altering or changing simulator elements. Most of the times, these files should be integrated with each other (e.g.
functionality for the aging of components requires the thermals of the same timestep)

## Contributing
For information about contributing to this project, see [CONTRIBUTING](CONTRIBUTING.md)
