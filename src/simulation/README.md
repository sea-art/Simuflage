## Simulator
Given a [designed embedded system](../design/README.md), we can run the simulator
to evaluate this design point. It is assumed that the design point is
created and stored in a variable. For example:
```python
dp = DesignPoint(components, applications, app_map)
```

### Creating and running the simulator
Once a design point is created, the simulator can be used in order to calculate the Time To Failure (TTF) 
of that design point. All that the simulator requires is a designpoint, which can be initialized as:
```python
from simulation import Simulator

sim = Simulator(dp)
sim.run()
```
Which will run the simulator until the given design point has failed. It will return the TTF of the given design point.

## Code structure
```
├───src
│   ├───design
│   ├───dse
│   └───simulation
│       ├───elements
│       └───faultmodels
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
about how to alter the simulator.


### Definitions
- **Simulation elements** - all the files located at ```src/simulation/elements/``` that desribe the
simulators behaviour. When changing the behavior of the simulator, files in these folders should be added or changed. The ```integrator```
file is used to make sure the elements integrate and work with each other.
- **Integrator** - since the aim of this simulator is to be modular and easily be changable, the simulator behavior
is expected to change frequently. The integrator file (located at ```src/simulation/integrator.py```) is used to specify
how the simulator should behave each timestep based on the ```simulator elements```.
- **Faultmodel** - This is the faultmodel that the system is using to determine when components are failing.
The current available faultmodel is [electromigration](https://en.wikipedia.org/wiki/Electromigration).

### Simulation
#### Adding simulation functionality
All elements of the simulator are currently:
- ```agings.py```
- ```components.py```
- ```thermals.py```

When adding elements to the simulator, it has to implement the abstract class ```simulator_element.py```.

### Integrator
The integrator, located at ```src/simulation/integrator.py``` is the file to edit the simulator functionality
upon altering or changing simulator elements. Most of the times, these files should be integrated with each other (e.g.
functionality for the aging of components requires the thermals of the same timestep). All logic regarding integration
of simulator elements should be defined here.
