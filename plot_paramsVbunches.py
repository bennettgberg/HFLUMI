import matplotlib.pyplot as plt

algo = "et"
nbunches = [302, 578, 2450]
param = [0.0 for n in nbunches]

for n,nb in enumerate(nbunches):
    fname = algo + "params_" + str(int(nb/100)) + "hb.txt"
    f = open(fname, "r")
    lastline = ""
    for l in f:
        lastline = l
    
    param[n] = float(lastline)

plt.plot(nbunches, param, '*')
plt.xlabel("Number of bunches")
plt.ylabel("Optimal Type 1 afterglow correction")
plt.show()
