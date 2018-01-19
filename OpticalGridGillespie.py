"""
    Determining blocking probabilities with the Gillespie method, as discussed in
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

import sys
import opticalgrid as og
from math import log
import getopt
from time import perf_counter
from datetime import timedelta
from scipy.special import ndtri as invstdnrmlcdf

# Parameters used in Section VII.C
m1 = 40
n2 = 4
rhos = [2, 10, 50]

min_batches = 5
max_batches = 50
batch_size = 4 * 10**7
conf_perc = 95
conf_fact = invstdnrmlcdf(1 - (100 - conf_perc) / 200)
phi = 1e-3

mu1 = 1
mu2 = mu1

text_file = "output.txt"

if len(sys.argv) < 2:
    print("No output file specified, using the default "+text_file)
else:
    try:
        opts, args = getopt.getopt(sys.argv[1:], "",
                                   ["m1=", "n2=", "rhos=", "rhommn=",
                                    "min-batches=", "max-batches=", "batch-size=",
                                    "conf-perc=", "out="])
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
        elif o == "--min-batches":
            min_batches = int(a)
        elif o == "--max-batches":
            max_batches = int(a)
        elif o == "--batch-size":
            batch_size = int(a)
        elif o == "--conf-perc":
            conf_perc = int(a)
            conf_fact = invstdnrmlcdf(1 - (100 - conf_perc) / 200)
        elif o == "--out":
            text_file = a

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
# Simulations
#################################################
policies = ['RA', 'LF', 'MF']

f = open(text_file, 'a')
print("SSA simulation with batch size {:,d},".format(batch_size)
      + " minimum {} batches and maximum {} batches.".format(
        min_batches, max_batches), file=f)
print("The relative error is computed with a"
      + " {}% confidence interval.".format(conf_perc))
print("Running simulations for rho = {}".format(rhos), file=f)
f.close()

og.print_header("mu_1 = {}, mu_2 = {}".format(mu1, mu2), text_file)


header = ["rho"]
for pol in policies:
    for bp in [1, 2]:
        header.extend([pol+":BP"+str(bp), pol+":BP"+str(bp)+"_relerr"])
f = open(text_file, 'a')
print(",".join(header), file=f)
f.close()

for rho in rhos:
    lambda1 = rho * mu1
    lambda2 = rho * mu2

    total_dur = perf_counter()

    row = [str(rho)]
    for pol in policies:
        # batch mean
        bp1, rel_err1, bp2, rel_err2, dur = og.gillespie(
            m1, n2, lambda1, lambda2, mu1, mu2, conf_fact, pol=pol,
            batch_size=batch_size, min_batches=min_batches,
            max_batches=max_batches, rel_accur=phi)
        row.extend([str(a) for a in
                    [bp1, rel_err1, bp2, rel_err2,]])
    f = open(text_file, 'a')
    print(",".join(row), file=f)
    f.close()
