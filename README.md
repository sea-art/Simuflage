
<p align="center">
    <a href="http://admorph.eu/">
        <img width="500" src="https://i.imgur.com/LJgU1Pf.png"/>
    </a>
</p>  

[![Build Status](https://travis-ci.com/sea-art/Simuflage.svg?token=N3rb3wFxBrspLC9Ysuz7&branch=master)](https://travis-ci.com/github/sea-art/Simuflage)
[![codecov](https://codecov.io/gh/sea-art/Simuflage/branch/master/graph/badge.svg?token=DJOIKL65KT)](https://codecov.io/gh/sea-art/Simuflage)
[![Repo Size](https://github-size-badge.herokuapp.com/sea-art/Simuflage.svg)](https://github.com/sea-art/Simuflage)
[![Python 3+7 ready](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/)
[![Licence](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

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
Which currently runs a Monte Carlo simulation on several random design points.

## Libraries used
- [Numpy](https://numpy.org/) - efficient multi-dimensional container of generic data 
- [Scipy](https://scipy.org/) - library used for scientific and technical computing

## Quickstart
- [Designing an embedded system](src/design/README.md)
- [Evaluate a single design point with the simulator](src/simulation/README.md)
- [Evaluating multiple design points with Monte Carlo](src/dse/README.md)

## Contributing
For information about contributing to this project, see [CONTRIBUTING](CONTRIBUTING.md)
