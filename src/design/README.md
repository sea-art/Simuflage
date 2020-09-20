## Design point


### Summary
A design point consists of:
- 1 or more [heterogeneous component(s)](#Components)
- 1 or more [application(s)](#Applications)
- A [mapping of applications to components](#Mapping-applications-to-components)
- An [adaptivity policy](#selecting-a-policy) (to handle component faults) 

This overview briefly describes how each of these elements of a 
design point can be created to make up a design point. 

### Import statements
```python
from design import Application
from design import Component
from design import DesignPoint
```

### Components
Components are the computing resources of an embedded system. They consist of two elements:
- a float representing the power capacity of a component,
- a location (x, y) where this component is positioned.

Each component has to be placed on a unique location (i.e. no two components can have the same position). 
The power capacity has to be a positive integer.

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

This will create a grid of components (represented in a 2D numpy float array as):
```python
[[0.,    0.],
 [120.,  100.]]
```
Where 0 represents no component at that coordinate.

### Applications
Applications can be mapped to components and will indicate how much of that component's 
capacity has to be spent to a mapped application. An application consists of:
- a float representing its power requirement.

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

### Selecting a policy
It is also possible to defined the adaptive policy that the system will use to remap applications when a component failure occurs.
At this moment, the current available policies are:
- random (randomly remap applications to non-failed components)
- least (maps applications to components with the least slack available)
- most (maps applications to components with the most slack available)

### Creating the designpoint
In order to create a design point the component, the applications and the mapping of 
applications to components are required. Once these are created, a design point is initialized as:
```python
dp = DesignPoint(components, applications, app_map, policy='random')
```
The default policy (when no policy is specified) is set to random, but it can also be specified as shown above.



### Quick designpoint creation
#### Manually
Instead of doing all previous steps manually, it is possible to create designpoints in a direct way, for example:
```python
DesignPoint.create(caps=[100, 120],
                   locs=[(1, 1), (0, 1)],
                   apps=[50, 40],
                   maps=[(0, 0), (1, 1)],
                   policy='random')
```
will do all previous steps in a single statement. All the arguments of this functions are corresponding index-wise. 

#### Randomly
It is also possibly to quickly generate a random DesignPoint
```python
dp = DesignPoint.create_random(n=4)
```
where ```n``` indicates the amount of components and applications that will randomly be created and mapped.
When n is not specified, it will be a random integer.
