import pandas as pd
import numpy as np
import math
from scipy.signal import find_peaks


#INPUTS----------------------------------------------------------
m1 = 1 #enter the first mode
m2 = 10 #enter the last mode
coev_file = "" #enter name of the file containing coevolution data
resnum = 580 #enter the number of residues
#----------------------------------------------------------------
t0 = 6
df = pd.read_excel(coev_file)
rawData = df.to_numpy()
Probthreshold = 0.80

#Connectivity matrix
NewKirchoff = np.zeros((resnum, resnum))

for k in range(len(rawData)):
    if rawData[k, 3] >= Probthreshold:
        row = int(rawData[k, 0]) - 1
        col = int(rawData[k, 1]) - 1
        val = -rawData[k, 2]

        NewKirchoff[row, col] = val
        NewKirchoff[col, row] = val

for i in range(resnum):
    for j in range(resnum):
        if abs(j - i) < 4 and i != j:
            NewKirchoff[i, j] = NewKirchoff[i, j] - 1

for i in range(resnum):
    NewKirchoff[i][i] = -np.sum(NewKirchoff[i,:])

U, S_vector, Vt = np.linalg.svd(NewKirchoff)
V = Vt.T
S = np.diag(S_vector)
winit = np.diag(S)
S1 = np.linalg.pinv(S)
w = np.diag(S1)

#Optimal Tau 1st Iteration
# 1. Define bounds first so calculations use the correct values
lower = 1
inc_t = 20
taubound = 4000

# 2. Calculate stable timeline grids
num_steps = math.ceil((taubound - lower) / inc_t) + 1
tau_range = np.linspace(lower, taubound, num_steps)

# Initialize tracking matrix
opttau_SeqGNM = np.zeros((resnum, resnum))

for i in range(0, resnum, 10):
    for j in range(4, resnum, 10):
        if i != j:
            Tij = np.zeros(num_steps)
            indx = -1

            for tau in np.arange(lower, taubound + inc_t, inc_t):
                indx += 1

                sum11 = 0
                sum12 = 0
                sum21 = 0
                sum23 = 0
                sum25 = 0

                for k in range(resnum - m2 - 1, resnum - m1):
                    sum11 += w[k] * V[j, k] * V[j, k]
                    sum12 += (w[k] * V[j, k] * V[j, k]) * math.exp(
                        -(winit[k] * tau) / t0
                    )
                    sum21 += w[k] * V[i, k] * V[i, k]
                    sum23 += w[k] * V[i, k] * V[j, k]
                    sum25 += (w[k] * V[i, k] * V[j, k]) * math.exp(
                        -(winit[k] * tau) / t0
                    )

                sum22 = sum11
                sum24 = sum12
                sum26 = sum25
                sum27 = sum23
                sum28 = sum11
                sum29 = sum12
                sum210 = sum21
                sum31 = sum11
                sum41 = sum21
                sum42 = sum11
                sum43 = sum23

                a = (sum11**2) - (sum12**2)
                b = (
                    (sum21 * sum22 * sum22)
                    + (2 * sum23 * sum24 * sum25)
                    - (((sum26**2) + (sum27**2)) * sum28)
                    - ((sum29**2) * sum210)
                )
                c = sum31
                d = sum41 * sum42 - sum43 * sum43

                if abs(a) < 1e-5 and a < 0:
                    a = abs(a)
                if abs(b) < 1e-5 and b < 0:
                    b = abs(b)
                if abs(c) < 1e-5 and c < 0:
                    c = abs(c)
                if abs(d) < 1e-5 and d < 0:
                    d = abs(c)  

                # Safety buffer added to handle tiny values close to zero without breaking
                comp1 = 0.5 * np.log(np.maximum(a, 1e-15))
                comp2 = 0.5 * np.log(np.maximum(b, 1e-15))
                comp3 = 0.5 * np.log(np.maximum(c, 1e-15))
                comp4 = 0.5 * np.log(np.maximum(d, 1e-15))

                Tij[indx] = comp1 - comp2 - comp3 + comp4

            # Standard peak detection
            indices, _ = find_peaks(Tij)
            indices = list(indices)

            if len(Tij) > 1 and Tij[0] > Tij[1]:
                indices.insert(0, 0)
            if len(Tij) > 1 and Tij[-1] > Tij[-2]:
                indices.append(len(Tij) - 1)

            pks = Tij[indices]
            locs = tau_range[indices]

            if len(locs) > 0:
                opttau_SeqGNM[i, j] = locs[0]

            del Tij

opt = opttau_SeqGNM

#Optimal Tau 2nd Iteration
opttau_SeqGNM_refined = np.zeros((resnum, resnum))

inc = 2

for i in range(0, resnum, 10):
    for j in range(4, resnum, 10):
        if i != j:

            if opt[i, j] - inc_t >= 0:
                lower = opt[i, j] - inc_t
                taubound = opt[i, j] + inc_t
            else:
                lower = 0.00001;
                taubound = opt[i, j] + inc_t

            num_steps = math.ceil((taubound - lower) / inc) + 1
            Tij = np.zeros(num_steps)
            tau_range = np.linspace(lower, taubound, num_steps)
            
            indx = -1

            for tau in np.arange(lower, taubound + inc, inc):
                indx += 1
                if indx >= num_steps: 
                    break 

                sum11 = 0
                sum12 = 0
                sum21 = 0
                sum23 = 0
                sum25 = 0

                for k in range(resnum - m2 - 1, resnum - m1):
                    sum11 += w[k] * V[j, k] * V[j, k]
                    sum12 += (w[k] * V[j, k] * V[j, k]) * math.exp(-(winit[k] * tau) / t0)
                    sum21 += w[k] * V[i, k] * V[i, k]
                    sum23 += w[k] * V[i, k] * V[j, k]
                    sum25 += (w[k] * V[i, k] * V[j, k]) * math.exp(-(winit[k] * tau) / t0)

                sum22 = sum11
                sum24 = sum12
                sum26 = sum25
                sum27 = sum23
                sum28 = sum11
                sum29 = sum12
                sum210 = sum21
                sum31 = sum11
                sum41 = sum21
                sum42 = sum11
                sum43 = sum23

                a = (sum11 ** 2) - (sum12 ** 2)
                b = (sum21 * sum22 * sum22) + (2 * sum23 * sum24 * sum25) - (((sum26 ** 2) + (sum27 ** 2)) * sum28) - ((sum29 ** 2) * sum210)
                c = sum31
                d = sum41 * sum42 - sum43 * sum43

                if abs(a) < 1e-5 and a < 0:
                    a = abs(a)
                if abs(b) < 1e-5 and b < 0:
                    b = abs(b)
                if abs(c) < 1e-5 and c < 0:
                    c = abs(c)
                if abs(d) < 1e-5 and d < 0:
                    d = abs(c) 

                comp1 = 0.5 * np.log(np.maximum(a, 1e-15))
                comp2 = 0.5 * np.log(np.maximum(b, 1e-15))
                comp3 = 0.5 * np.log(np.maximum(c, 1e-15))
                comp4 = 0.5 * np.log(np.maximum(d, 1e-15))

                Tij[indx] = comp1 - comp2 - comp3 + comp4

            indices, _ = find_peaks(Tij)
            indices = list(indices)

            if len(Tij) > 1 and Tij[0] > Tij[1]:
                indices.insert(0, 0)
            if len(Tij) > 1 and Tij[-1] > Tij[-2]:
                indices.append(len(Tij) - 1)

            locs = tau_range[indices]

            if len(locs) > 0:
                opttau_SeqGNM_refined[i, j] = locs[0]

            del Tij

#GNM-TE
Tij = np.zeros((resnum, resnum))
net = np.zeros((resnum, resnum))

# 2. Calculate the average tau scaling factor (handling MATLAB's full-matrix math)
# MATLAB: sum(sum(matrix)) gets the grand total of all entries
nonzero_values = opttau_SeqGNM[opttau_SeqGNM > 0]
 
tau = np.mean(nonzero_values)*3

# 3. Main Transfer Entropy Matrix Loop
for i in range(0, resnum):
    for j in range(0, resnum):
        if i == j:
            Tij[i, j] = 0
        else:
            sum11 = 0
            sum12 = 0
            sum21 = 0
            sum23 = 0
            sum25 = 0

            # Mode summation loop
            for k in range(resnum - m2 - 1, resnum - m1):
                sum11 += w[k] * V[j, k] * V[j, k]
                sum12 += (w[k] * V[j, k] * V[j, k]) * math.exp(
                    -(winit[k] * tau) / t0
                )
                sum21 += w[k] * V[i, k] * V[i, k]
                sum23 += w[k] * V[i, k] * V[j, k]
                sum25 += (w[k] * V[i, k] * V[j, k]) * math.exp(
                    -(winit[k] * tau) / t0
                )

            sum22 = sum11
            sum24 = sum12
            sum26 = sum25
            sum27 = sum23
            sum28 = sum11
            sum29 = sum12
            sum210 = sum21
            sum31 = sum11
            sum41 = sum21
            sum42 = sum11
            sum43 = sum23

            a = (sum11**2) - (sum12**2)
            b = (
                (sum21 * sum22 * sum22)
                + (2 * sum23 * sum24 * sum25)
                - (((sum26**2) + (sum27**2)) * sum28)
                - ((sum29**2) * sum210)
            )
            c = sum31
            d = sum41 * sum42 - sum43 * sum43

            # Precision limit adjustments matching MATLAB's thresholds
            if abs(a) < 1e-5 and a < 0:
                a = abs(a)
            if abs(b) < 1e-5 and b < 0:
                b = abs(b)
            if abs(c) < 1e-5 and c < 0:
                c = abs(c)
            if abs(d) < 1e-5 and d < 0:
                d = abs(c)  # Kept strict 'abs(c)' legacy match

            comp1 = 0.5 * np.log(np.maximum(a, 1e-15))
            comp2 = 0.5 * np.log(np.maximum(b, 1e-15))
            comp3 = 0.5 * np.log(np.maximum(c, 1e-15))
            comp4 = 0.5 * np.log(np.maximum(d, 1e-15))

            Tij[i, j] = comp1 - comp2 - comp3 + comp4

# 4. Net Information Transfer Directionality Loop
for i in range(0, resnum):
    for j in range(0, resnum):
        net[i, j] = Tij[i, j] - Tij[j, i]

#COLLECTIVITY

alfa1 = np.zeros(resnum)
col1 = np.zeros(resnum)

for i in range(0, resnum):
    # Extract row i and filter for strictly positive elements
    a = net[i, :]
    a_pos = a[a > 0]
    
    if len(a_pos) > 0:
        alfa1[i] = 1.0 / np.sum(a_pos ** 2)
        
        insidesum = 0.0
        for j in range(0, len(a_pos)):
            val = alfa1[i] * (a_pos[j] ** 2)
            # Use np.maximum to protect against log(0)
            inside = val * np.log(np.maximum(val, 1e-15))
            insidesum += inside
            
        col1[i] = math.exp(-1.0 * insidesum) / resnum
    else:
        alfa1[i] = 0.0
        col1[i] = 0.0  # Handle residues with no positive outgoing net tracking

#  netp and netp_avg Calculation 
netp = np.copy(net)
count = 1 
summ_netp = np.zeros((resnum, 1))
netp_avg = np.zeros((resnum, 1))

for i in range(0, resnum):
    sum_netp = 0.0
    for j in range(0, resnum):
        summ_netp[i, 0] = sum_netp + netp[i, j]
        sum_netp = summ_netp[i, 0]
        
    # Since count stays 1, this effectively acts as a row sum matrix assignment
    netp_avg[i, 0] = summ_netp[i, 0] / count