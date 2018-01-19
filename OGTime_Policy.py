"""
    Determining policy-dependent bounds on the blocking probabilities, as discussed in
    (Erreygers et al, 2018).

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
from time import perf_counter
from datetime import timedelta
import getopt

# Parameters used in Section VII.C
m1 = 40
n2 = 4
rhos = [2, 10, 50]

maxit = 10**6
reps = 5

mu1 = 1
mu2 = mu1

text_file = "output.txt"

if len(sys.argv) < 2:
    print("No output file specified, using the default "+text_file)
else:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "",
                                   ["m1=", "n2=", "rhos=",
                                    "rhommn=", "maxit=", "out=", "repeat="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        sys.exit(2)

    for o, a in opts:
        if o == "--m1":
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
        elif o == "--maxit":
            maxit = int(a)
        elif o == "--out":
            text_file = a
        elif o == "--repeat":
            reps = a

print("Output is written out to {}".format(text_file))

tic = perf_counter()
stateSpace = og.StateSpace(m1, n2, reduced=True)
toc = timedelta(seconds=perf_counter() - tic)

og.print_header(
    "m1 = {}, n2 = {}".format(m1, n2), text_file)

f = open(text_file, 'a')
print("Generating the state space took {}".format(toc), file=f)
f.close()

#################################################
# Looping the computations over all possible values of rho
#################################################
ULs = ['L', 'U']  # 'L' and/or 'U'
BPs = [1, 2]  # 1 and/or 2
policies = ['R', 'LM']  # 'R' and/or 'LM'

f = open(text_file, 'a')
print("Using the partially imprecise chain with the policies {}.".format(
    ', '.join(policies)), file=f)
print("The state space has {} states".format(stateSpace.dim), file=f)
print("Running {} simulations for rho = {}".format(reps, rhos), file=f)
print("Maximum number of iterations is {}".format(maxit), file=f)
print("mu_1 = {}, mu_2 = {}".format(mu1, mu2), file=f)
print("", file=f)
f.close()


og.print_header("mu_1 = {}, mu_2 = {}".format(mu1, mu2), text_file)
tableHeader = ["rho"]
for pol in policies:
    for i in BPs:
        for ul in ULs:
            base = pol+":BP"+str(i)+ul
            tableHeader.extend([base+"_t_su", base+"_t_c"])
tableHeader.append("total_t")
f = open(text_file, 'a')
print(",".join(tableHeader), file=f)
f.close()

for rho in rhos:
    lambda1 = rho * mu1
    lambda2 = rho * mu2

    row = [str(rho)]

    total_dur = perf_counter()

    for pol in policies:
        dur_su = perf_counter()
        apply_ltro, normQ = stateSpace.construct_ltro(
            mu1, mu2, lambda1, lambda2, pol=pol, impr=True)
        dur_su = timedelta(seconds=perf_counter() - dur_su)
        # Usually 2 / normQ, but we go for a safe margin
        delta = .5 / normQ
        phi = 1e-3

        for bp in BPs:
            if bp is 1:
                gamble = np.copy(stateSpace.g_BP1)
            elif bp is 2:
                gamble = np.copy(stateSpace.g_BP2)
            for ul in ULs:
                rep_dur_c = timedelta(seconds=0)
                for _ in range(reps):
                    # print("Rho = {}, {}, BP{}, {}.".format(
                    #     rho, pol, bp, "lower" if ul == 'L' else "upper"))
                    BP, act_phi, numit, dur_c = og.empirical(
                        apply_ltro, gamble, delta, phi, ul, maxit=maxit)
                    rep_dur_c += dur_c
                row.extend([str(dur_su), str(rep_dur_c / reps)])

    total_dur = timedelta(seconds=(perf_counter() - total_dur) / reps)
    row.append(str(total_dur))
    f = open(text_file, 'a')
    print(','.join(row), file=f)
    f.close()
