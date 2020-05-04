## Simulator
Given a designed embedded system, we can run the simulator
to evaluate this design point. It is assumed that the design point is
created and stored in a variable as:
```python
dp = Designpoint(components, applications, app_map)
```

### Creating and running the simulator
Once a design point is created, the simulator can be used in order to calculate the Time To Failure (TTF) 
of that design point. All that the simulator requires is a designpoint, which can be initialized as:
```python
from simulation.simulator import Simulator

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
