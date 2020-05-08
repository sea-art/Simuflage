## Monte Carlo evaluation
It is possible to evaluate a list of design points via a Monte Carlo simulation to rank these based on MTTF.
A list of ```n``` random design points with ```k``` components can be created as:
```python
designpoints = [DesignPoint.create_random(k) for _ in range(n)]
```
Given this list of designpoints, we can evaluate these via:
```python
monte_carlo(designpoints, iterations=10000, parallelized=True)
```
It is advised to use the parallelized version of the Monte Carlo simulation for performance purposes.

## Example
The following example will create an ```n x n``` grid of homogeneous processors with a power capacity of 100 and 
will print the MTTF of this Design Point per every 10% workload increase.
```python
import numpy as np

from design.designpoint import DesignPoint
from design.mapping import all_possible_pos_mappings
from dse.montecarlo import monte_carlo

dps = []
n = 2

for i in range(1, 11):
    dps.append(DesignPoint.create(caps=np.repeat(100, n * n),
                                  locs=all_possible_pos_mappings(n * n),
                                  apps=np.repeat(i * 10, n * n),
                                  maps=[(i, i) for i in range(n * n)],
                                  policy='random'))

results = monte_carlo(dps, iterations=len(dps) * 1000)

for k, v in results.items():
    print("Workload:", (k + 1) / 10, "\tMTTF:", np.around(v, 1), "\t(Years: " + str((v / (24 * 365))) + ")")
```
Which will generate the output:
```bash
Running montecarlo simulation
Total iterations:                   10000
Iterations per design point:        1000 

Workload: 0.1 	MTTF: 646803.7 	(Years: 73.83603493150684)
Workload: 0.2 	MTTF: 443142.8 	(Years: 50.58708321917808)
Workload: 0.3 	MTTF: 305109.0 	(Years: 34.82979668949771)
Workload: 0.4 	MTTF: 223972.1 	(Years: 25.567592694063926)
Workload: 0.5 	MTTF: 162715.2 	(Years: 18.57479360730594)
Workload: 0.6 	MTTF: 108100.0 	(Years: 12.340188242009132)
Workload: 0.7 	MTTF: 82692.8 	(Years: 9.439820662100457)
Workload: 0.8 	MTTF: 63780.5 	(Years: 7.280879680365297)
Workload: 0.9 	MTTF: 49581.9 	(Years: 5.660037100456622)
Workload: 1.0 	MTTF: 38834.8 	(Years: 4.4331950913242)

Process finished with exit code 0
```