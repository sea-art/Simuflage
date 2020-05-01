This file explains the structure of this project in more detail aiming to provide information
about how to contribute/alter the simulator.


## Definitions
- **Simulation elements** - all the files located at ```src/simulation/elements/``` that desribe the
simulators behaviour. When changing the behavior of the simulator, files in these folders should be added or changed. The ```integrator```
file is used to make sure the elements integrate and work with each other.
- **Integrator** - since the aim of this simulator is to be modular and easily be changable, the simulator behavior
is expected to change frequently. The integrator file (located at ```src/simulation/integrator.py```) is used to specify
how the simulator should behave each timestep based on the ```simulator elements```.

## Simulation
### Adding simulation functionality
All elements of the simulator are currently:
- ```agings.py```
- ```components.py```
- ```thermals.py```

When adding elements to the simulator, it has to extend the abstract class ```simulator_element.py```.

## Integrator
The integrator, located at ```src/simulation/integrator.py``` is the file to edit the simulator functionality
upon altering or changing simulator elements.