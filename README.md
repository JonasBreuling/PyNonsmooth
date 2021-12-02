# nonsmooth_mech_benckmarks

## Table of Contents

- [nonsmooth_mech_benckmarks](#nonsmooth_mech_benckmarks)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Simulate a benchmark](#simulate-a-benchmark)
  - [Mechanics](#mechanics)

## Requirements
* Python 3.x (Tested on Python 3.9.9 for macOX)
* Python packages:
    * numpy
    * matplotlib
    * tqdm (used to include progress bar)

## Simulate a benchmark
This benchmark collection is organized as follows: 
The folder `solvers` contains a collection of solvers, which can be used to simulate the benchmarks. The folder `benchmarks` contains a folder for each benchmark examples, e.g. the folder `benchmarks/bouncingball` for the rotating bouncing ball example. Each benchmark folder contains the file defining the benchmark system, e.g. `bouncing_ball_system.py`. The 'system' file defines the class containing all system properties and providing the mass matrix, forces, contact kinematics, etc. as methods.
Finally the acutal simulation is done by running the 'scenario' files, e.g. `bouncing_ball_scenario1.py`, which sets the initial conditions, calls the solver and plots the results. For instance, a simulation of the bouncing ball can be performed by calling
```bash
python benchmarks/bouncing_ball/bouncing_ball_scenario1.py
```

## Mechanics

The most general structure of governing equations, which is able to describe any of our benchmarks is given by the following. This is also the nomenclature used in the code.

*Kinematic variables*: time $`t`$, gen. coordinates $`q`$, gen. velocitites $`u`$, gen. accelerations $`a`$.

*Kinetic equation*
```math
M(t,q)\ du = h(t, q, u)\ dt + W_g(t, q)\ dP_g + W_\gamma(t, q)\ dP_\gamma + W_N(t, q)\ dP_N + W_F(t, q)\ dP_F
```
*Kinematic equation*
```math
\dot{q}(t, q, u) = B(t, q)\ u + \beta(t, q)
```
*Bilateral constraints*
```math
g(t, q) = 0
```
```math
\gamma(t, q, u) = 0
```
*Contact laws*: Normal cone inclusions linking gaps $`g_N(t, q)`$ and friction velcities $`\gamma_F(t, q, u)`$ to percussion measures $`dP_N`$ and $`dP_F`$, respectively.

This nomenclature is based on 
>**A nonsmooth generalized-alpha method for mechanical systems with frictional contact**
>
>Giuseppe Capobianco, Jonas Harsch, Simon R. Eugster and Remco I. Leine
>
>*Int J Numer Methods Eng. 2021; 122: 6497– 6526. https://doi.org/10.1002/nme.6801*
