## Design point

### Import statements
```python
from design.application import Application
from design.component import Component
from design.designpoint import Designpoint
```

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

### Quick designpoint creation
#### Manually
Instead of doing all previous steps manually, it is possible to create designpoints in a direct way
```python
Designpoint.create_designpoint(capacities=[100, 120], 
                               locs=[(1, 1), (0, 1)],
                               apps=[50, 40], 
                               mapping=[(0, 0), (1,1)])
```
will do all previous steps in a single statement. All the arguments of this functions are corresponding index-wise. 

#### Randomly
It is also possibly to quickly generate a random Designpoint
```python
dp = Designpoint.create_random(n)
```
where ```n``` indicates the amount of components and applications that will reandomly be created and mapped.
