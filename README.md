<p align="center">
    <img width="500" src="https://i.imgur.com/MF0plyC.png"/>
</p>  

<p align="center">
    <img width="400" src="https://i.imgur.com/LJgU1Pf.png"/>
</p>  


## Summary
This repository contains a simple simulator that can be used to receive time to failures of a 
given design point.

## About

## Running
```bash
git clone git@github.com:sea-art/DSE_simulator.git
cd DSE_simulator
python3 main.py
```

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

Similar to the components, all applications have to be bundled together as well:
```python
applications = [a1, a2]
```

### Mapping applications to components
Applications have to be mapped to components, and can be done as follows:
```python
app_map = [(c1, a1), (c2, a2)]
```
Which maps the application of 50 to component 1, and the appliaction of 40 to component 2.

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

## Contributing

## Code structure

## Contributors
Siard Keulen