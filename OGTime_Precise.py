"""
    Determining (approximations of) the blocking probabilities iteratively,
    as discussed in (Erreygers et al, 2018).

    Copyright (C) 2018 Alexander Erreygers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from math import log
import sys
import opticalgrid as og
import getopt
from time import perf_counter
from datetime import timedelta

# Parameters used in Section VII.C
m1 = 40
n2 = 4
rhos = [2, 10, 50]

use_lgmres = False

reduced_st_sp = False

text_file = 'output.txt'

maxit = 10**6
reps = 5

mu1 = 1
mu2 = mu1

if len(sys.argv) < 2:
    print("No output file specified, using the default "+text_file)
else:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "dr",
                                   ["detailed", "reduced", "m1=", "n2=",
                                    "rhos=", "rhommn=", "method=",
                                    "maxit=", "out=", "repeat="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        sys.exit(2)

    for o, a in opts:
        if o in ("-d", "--detailed"):
            reduced_st_sp = False
        elif o in ("-r", "--reduced"):
            reduced_st_sp = True
        elif o == "--m1":
            m1 = int(a)
        elif o == "--n2":
            n2 = int(a)
        elif o == "--rhos":
            rhos = [float(r) for r in a.split(',')]
        elif o == "--rhommn":
            rhos = a.split(',')
            rhomin = float(rhos[0])
            rhomax = float(rhos[1])
            numrhos = int(rhos[2])
            # One way to get evenly spaced x-axis in a logarithmic plot
            if numrhos is 2:
                rhos = [rhomin, rhomax]
            else:
                logmin = log(rhomin, 10)
                logmax = log(rhomax, 10)
                rhos = [10 ** (logmin + (logmax - logmin) * _i / (numrhos-1))
                        for _i in range(numrhos)]
        elif o == "--method":
            if a == "lgmres":
                use_lgmres = True
        elif o == "--maxit":
            maxit = int(a)
        elif o == "--out":
            text_file = a
        elif o == "--repeat":
            reps = a

print("Output is written out to {}".format(text_file))

tic = perf_counter()
stateSpace = og.StateSpace(m1, n2, reduced=reduced_st_sp)
toc = timedelta(seconds=perf_counter() - tic)

f = open(text_file, 'a')
print("Generating the state space took {}".format(toc), file=f)
f.close()

og.print_header(
    "m1 = {}, n2 = {}".format(m1, n2), text_file)

#################################################
# Actual computations
#################################################
f = open(text_file, 'a')
print("Using the precise chain", file=f)
print("Using the {} approximation method".format(
    'LGMRES' if use_lgmres else
    'iterative'), file=f)
print("Using the {} state space".format(
    'reduced' if reduced_st_sp else 'detailed'), file=f)
print("The state space has {} states".format(stateSpace.dim), file=f)
print("Running {} simulations for rho = {}".format(reps, rhos), file=f)
print("mu_1 = {}, mu_2 = {}".format(mu1, mu2), file=f)
print("", file=f)
f.close()

if not reduced_st_sp:
    policies = ['RA', 'LF', 'MF']
    # policies = ['RA']
else:
    policies = ['R', 'LM']
    # policies = ['LM']

BPs = [1, 2]

header = ["rho"]
if use_lgmres:
    for pol in policies:
        base = pol+":BP"
        header.extend([pol+":SS_t_su", pol+":SS_t_c"])
else:
    for pol in policies:
        for bp in BPs:
            base = pol+":BP"+str(bp)
            header.extend([base+"_t_su", base+"_t_c"])
header.append("total_t")
f = open(text_file, 'a')
print(",".join(header), file=f)
f.close()

for rho in rhos:
    lambda1 = rho * mu1
    lambda2 = rho * mu2
    total_dur = perf_counter()

    row = [str(rho)]
    if use_lgmres:
        for pol in policies:
            rep_dur_su = timedelta(seconds=0)
            rep_dur_c = timedelta(seconds=0)
            for _ in range(reps):
                dur_su1 = perf_counter()
                Q, normQ = stateSpace.construct_Q_matrix(
                        mu1, mu2, lambda1, lambda2, pol=pol)
                dur_su1 = timedelta(seconds=perf_counter() - dur_su1)

                bp1, bp2, dur_su2, dur_c = og.steady_state_numerical(
                    Q, stateSpace.g_BP1, stateSpace.g_BP2, tol=1e-9,
                    method='ILU-LGMRes')
                rep_dur_su += dur_su1 + dur_su2
                rep_dur_c += dur_c
            row.extend([str(rep_dur_su / reps), str(rep_dur_c / reps)])
    else:
        for pol in policies:
            dur_su = perf_counter()
            apply_ltro, normQ = stateSpace.construct_ltro(
                mu1, mu2, lambda1, lambda2, pol=pol, impr=False)
            dur_su = timedelta(seconds=perf_counter() - dur_su)
            delta = .9 * 2 / normQ
            phi = 1e-3
            for bp in BPs:
                if bp is 1:
                    gamb = np.copy(stateSpace.g_BP1)
                elif bp is 2:
                    gamb = np.copy(stateSpace.g_BP2)
                rep_dur_c = timedelta(seconds=0)
                for _ in range(reps):
                    BP, act_phi, numi, dur_c = og.empirical(
                        apply_ltro, gamb, delta, phi, 'P', maxit=maxit)
                    rep_dur_c += dur_c
                row.extend([str(dur_su), str(rep_dur_c / reps)])

    row.append(str(timedelta(seconds=(perf_counter() - total_dur) / reps)))
    f = open(text_file, 'a')
    print(",".join(row), file=f)
    f.close()
