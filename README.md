<p align="center">
    <img width="500" src="https://i.imgur.com/MF0plyC.png"/>
</p>  
<br/>
<p align="center">
    <img width="400" src="https://i.imgur.com/LJgU1Pf.png"/>
</p>  


## Summary
This repository contains a simple simulator that can be used to receive time to failures of a 
given design point.

## About
This repository complements a thesis project of the MSc Software Engineering master at the University of Amsterdam. 
This project falls into the scope of [ADMORPH](http://admorph.eu/).

## Running
```bash
git clone git@github.com:sea-art/DSE_simulator.git
cd DSE_simulator
python3 main.py
```

## Project structure
```
src
+-- design
|   +-- application.py
|   +-- component.py
|   +-- designpoint.py
|   +-- mapping.py
+-- simulation
|   +-- agings.py
|   +-- components.py
|   +-- simulator.py
|   +-- thermals.py
```
All the code about *design points* are positioned in the ```src/design/``` folder.<br>
 
All the code regarding the *simulation* are placed in the ```src/simulation/``` folder.<br>
The ```simulator.py``` can be seen as the *main* function for the simulator. All other files 
(i.e. ```agings.py```, ```components.py``` and ```thermals.py```) provide an ```iterate()``` function
based on some parameters, which the simulator will call each simulation iteration.

## Example

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
sim.run(until_failure=True, debug=False)
```
When running the simulator it is either possible to run a fixed amount of iterations, 
or to keep running the simulator until a failure has occurred.

## Contact
Don't hesitate to contact me regarding any questions about the code or project!<br>

## Contributors
Siard Keulen