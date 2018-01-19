# Imprecise Markov Models for Scalable and Robust Performance Evaluation of Flexi-Grid Spectrum Allocation Policies

This repository contains the Python 3 and C code that was used to run the numerical simulations reported in (Erreygers et al., 2018).

## opticalgrid.py
The `opticalgrid` module contains methods to generate the (imprecise) continuous-time Markov chain models introduced in (Erreygers et al., 2018).
It also contains Python (or C) implementations of the various numerical methods discussed in (Erreygers et al., 2018), which can be used to to determine blocking probabilities.

The module imports [NumPy](http://www.numpy.org/) and [SciPy](https://scipy.org/) for easy vector manipulations. Both of these packages should be installed in order for the package to work. 
In order to (efficiently) run the Gillespie method, one should compile the [gillespy.c](gillespy.c) file for use with Python 3.
Executing the command 
```gcc -shared -fPIC -Wall -O3 -funroll-loops gillespy.c -I/usr/include/python3.5 -o gillespy.so```
should result in a `gillespy.so` file that can be imported by Python.
If executing this command is unsuccessful, you can resort to the [gillespy.so](gillespy.so) file in this repository, which was compiled on a Linux machine with a 64-bit Intel processor.

## Numerical experiments
The various numerical experiments that are reported in (Erreygers et al., 2018) can be obtained by running the bash scripts.
The raw `.txt` output of our simulations can be found in the [raw_results/](raw_results/) folder, for more cleaned up data one should consult (Erreygers et al., 2018).

## References
1. Alexander Erreygers, Cristina Rottondi, Giacomo Verticale and Jasper De Bock. _''Imprecise Markov Models for Scalable and Robust Performance Evaluation of Flexi-Grid Spectrum Allocation Policies''_. [arXiv:1801.05700](https://arxiv.org/abs/1801.05700) [cs.NI].