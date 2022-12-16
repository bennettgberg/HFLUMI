
def get_hfsbr(fname):
    newf = open(fname, "r")
    HFSBR = newf.read().split(',')
    newf.close()
    for i in range(len(HFSBR)):
        HFSBR[i]=float(HFSBR[i])
    return HFSBR

mode = "ET"
#mode = "OC"

if mode == "ET":
    old_file = "HFSBR_ET_22v0.txt"
    #new_file = "HFSBR_ET_22v1_2.txt"
    new_file = "HFSBR_ET_22v11.txt"
else:
    old_file = "HFSBR_OC_22v0.txt"
    #new_file = "HFSBR_OC_22v3.txt"
    #new_file = "HFSBR_OC_22v3_1.txt"
    new_file = "HFSBR_OC_22v11.txt"

HFSBR_old = get_hfsbr(old_file)
HFSBR_new = get_hfsbr(new_file)

bcid = [j for j in range(len(HFSBR_old))] 

import matplotlib.pyplot as plt
plt.plot(bcid, HFSBR_old, '-', label="Old HFSBR")
plt.plot(bcid, HFSBR_new, '-', label="New HFSBR")
for nb in ["302b", "578b", "2450b"]:
    bfile = "HFSBR_" + mode + "_" + nb + "_v11.txt"
    HFSBR_b = get_hfsbr(bfile)
    plt.plot(bcid, HFSBR_b, '-', label=nb) 
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.xlabel("BCID")
plt.ylabel("Afterglow fraction")

plt.show()
