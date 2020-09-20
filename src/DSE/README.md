## Monte Carlo evaluation
It is possible to evaluate a list of design points via a Monte Carlo simulation to rank these based on MTTF.
A list of ```n``` random design points with ```k``` components can be created as:
```python
designpoints = [DesignPoint.create_random() for _ in range(n)]
```
Given this list of designpoints, we can evaluate these via:
```python
from DSE import monte_carlo

monte_carlo(designpoints, sample_budget=10000, parallelized=True)
```
It is advised to use the parallelized version of the Monte Carlo simulation for performance purposes.

## Example
The following example will create an ```n x n``` grid of homogeneous processors with a computational capability of 100 
and will print the MTTF of this Design Point per every 10% workload increase.
```python
import numpy as np

from DSE import monte_carlo
from design import DesignPoint
from design.mapping import all_possible_pos_mappings

dps = []
n = 3

for i in range(1, 11):
    dps.append(DesignPoint.create(caps=np.repeat(100, n * n),
                                  locs=all_possible_pos_mappings(n * n)[:n*n],
                                  apps=np.repeat(i * 10, n * n),
                                  maps=[(i, i) for i in range(n * n)],
                                  policy='random'))

results = monte_carlo(dps, sample_budget=len(dps) * 1000)

for k, v in results.items():
    print("Workload: {}\t\tMTTF: {:.4f} years\t\tAvg. Power usage: {:.2f}".format((k + 1) / 10,
                                                                                  v[0] / (24 * 365),
                                                                                  v[1]))
```
Which will generate the output:
```bash
Workload: 0.1		MTTF: 73.8746 years		Avg. Power usage: 137.80
Workload: 0.2		MTTF: 46.3514 years		Avg. Power usage: 203.20
Workload: 0.3		MTTF: 29.5515 years		Avg. Power usage: 269.28
Workload: 0.4		MTTF: 20.3000 years		Avg. Power usage: 332.89
Workload: 0.5		MTTF: 13.9152 years		Avg. Power usage: 396.56
Workload: 0.6		MTTF: 8.5410 years		Avg. Power usage: 464.37
Workload: 0.7		MTTF: 6.1844 years		Avg. Power usage: 527.77
Workload: 0.8		MTTF: 4.5329 years		Avg. Power usage: 591.17
Workload: 0.9		MTTF: 3.3609 years		Avg. Power usage: 654.56
Workload: 1.0		MTTF: 2.5191 years		Avg. Power usage: 717.96
```