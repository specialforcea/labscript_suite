import numpy as np 
import visa
import time
#table = np.multiply(np.ones((3,110),dtype = int),20)

#map list, from number of wire to its SDIN position.
#map_list = [1,8,3,4,17,24,19,14,21	,22,23,18,13,20,15,16,29,36,31,26,	33,	34,	35,30,
#           25,32	,27,28,41,48,43,38,45,46,47,42,37,44,39,40,53,60,55,50,57,58,59,54,
#           49,56	,51,52,65,72,67,62,69,70,71,66,61,68,63,64,77,84,79,74,81,82,83,78,
#           73,80	,75,76,89,96,91,86,93,94,95,90,85,92,87,88]
#
#ground_list = [2,	5,6,7,9,10,11,12]






map_list = [12	,5	,10	,9	,20	,13	,18	,23, 
            16	,15	,14	,19	,24	,17	,22	,21,
            32	,25	,30	,35	,28	,27	,26	,31, 
            36	,29	,34	,33	,44	,37	,42	,47, 
            40	,39	,38	,43	,48	,41	,46	,45,
            56	,49	,54	,59	,52	,51	,50	,55,
            60	,53	,58	,57	,68	,61	,66	,71,
            64	,63	,62	,67	,72	,65	,70	,69,
            80	,73	,78	,83	,76	,75	,74	,79,
            84	,77	,82	,81	,92	,85	,90	,95,
            88	,87	,86	,91	,96	,89	,94	,93]


ground_list = [1,2,3,4,6,7,8,11]


n = 4 #group number

def set_group(n):
	table = np.zeros((n,108),dtype = int)
	return table

table = set_group(n)


div = np.zeros((n,6),dtype = int)
phase_delay = np.zeros((n,6),dtype = int)
nmbr = np.zeros((n,96),dtype = int)

#def set_N(N,table,n):
#	for i in range(n):
#		table[i,0] = N[i]
#
#	return table

#
#def set_M(M,table,n):
#	for i in range(n):
#		table[i,1] = M[i]
#
#	return table


def set_div(C,table,n):
	for i in range(n):
		table[i,0:6] = C[i]
	return table

def set_phase_delay(phase_delay,table,n):
	for i in range(n):
		table[i,6:12] = phase_delay[i]
	return table

def set_nmbr(nmbr,table,n):
	for i in range(n):
		table[i,12:108] = nmbr[i]
        for j in range(8):
            table[i,11+ground_list[j]] = 0
	return table

def mask_ground(nmbr,ground_list):
    for i in range(8):
        for j in range(n):
             nmbr[j,ground_list[i]-1] = 0
    return nmbr
N = np.array([1,1,1,1])
M = np.array([3,3,3,3])
div = np.ones((n,6),dtype = int)*4

sf = np.array([0,1,2,3,4,5])
one = np.ones(6 ,dtype = int)
phase_delay = np.outer(one,sf)


nmbr[0,:] = np.ones((1,96),dtype = int)*0
nmbr[1,:] = np.ones((1,96),dtype = int)*0
nmbr[2,:] = np.ones((1,96),dtype = int)*0
nmbr[3,:] = np.ones((1,96),dtype = int)*0

#nmbr = mask_ground(nmbr,ground_list)
#table = set_N(N,table,n)
#table = set_M(M,table,n)
table = set_div(div,table,n)
table = set_phase_delay(phase_delay,table,n)
table = set_nmbr(nmbr,table,n)









np.save('C:/Experiments/labscript_shared_drive/Experiments/rb_chip/table.npy',table)





