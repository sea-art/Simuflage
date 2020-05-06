## Monte Carlo evaluation
It is possible to evaluate a list of design points via a Monte Carlo simulation to rank these based on MTTF.
A list of ```n``` random design points with ```k``` components can be created as:
```python
designpoints = [Designpoint.create_random(k) for _ in range(n)]
```
Given this list of designpoints, we can evaluate these via:
```python
monte_carlo(designpoints, iterations=10000, parallelized=True)
```
It is advised to use the parallelized version of the Monte Carlo simulation for performance purposes.
