import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle

argv = sys.argv

# provide detailed results
with open(argv[1],'rb') as fid:
	details_dict = pickle.load(fid)

#print(details_dict.keys())
ibcurve_arr = [] # put necessary elements for IB curve only
for k,v in details_dict.items():
	#print(v)
	tmp_ib_set = [v['beta'],float(k),v['IZY']]
	ibcurve_arr.append(tmp_ib_set)

ibcurve_arr = sorted(ibcurve_arr,key=lambda x:x[0])
ibcurve_arr = np.array(ibcurve_arr)
#print(ibcurve_arr)

fig,ax = plt.subplots()
ax.grid('on')
ax.set_xlabel(r"$I(Z;X)$ (nats)")
ax.set_ylabel(r"$I(Z;Y)$ (nats)")
ax.scatter(ibcurve_arr[:,1],ibcurve_arr[:,2],label="solution",marker='o',color='tab:red')

plt.show()
