# nonsmooth_mech_benckmarks

The most general structure of governing equations, which is able to describe any of our benchmarks is given by the following.

*Kinematic variables*: time $`t`$, gen. coordinates $`q`$, gen. velocitites $`u`$, gen. accelerations $`a`$.

*Kinetic equation*
```math
M(t,q)\ du = h(t, q, u)\ dt + W_g(t, q)\ dP_g + W_\gamma(t, q)\ dP_\gamma + W_N(t, q)\ dP_N + W_F(t, q)\ dP_F
```
*Kinematic equation*
```math
\dot{q}(t, q, u) = B(t, q)\ u + \beta(t, q)
```
*Ideal bilateral constraints*
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
