'''
Mod from data_gen_from_PTx.py from nC4-nC10 data gen from PTx larger

Add random_comp_nb
Add LHS (todo)

Component-specific data sets, do one at a time.
Components from Monahans Clearfork
Generate training instances given a range of P, T. Use LHS or random sampling for the x.
Discard multiple positive real roots.

Must decide num_P and num_x. We want to show 3-phase behavior. So just limit the P range to there.
Use isothermal T = 90 F, 305.372 K.
P = [1000,2000] psia
Try num_P = 1000
num_x = 1000
10^6 instances max (should be easy to fit)

'''

from Stability_and_flash import pr
import numpy as np
import csv
import os
import datetime
from numba import jit
from pyDOE import *         # For LHS

# Writes data
def data_writer(datafilename, a_mix, b_mix, b_i, sum_xiAij, ln_phi_x):
    b_i_max = np.max(b_i)
    templist = [a_mix, b_mix, b_i, sum_xiAij, ln_phi_x]

    # datafilename is defined at initiation
    with open(datafilename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(templist)
    return

# Returns all samples at once (required)
#@jit(nopython=True)
def lhs_simplex(n_comps, n_samples):
    temp = lhs(n_comps-1, samples=n_samples, criterion='c')
    temp = np.sort(temp, axis=1)
    arr = np.empty((n_samples, n_comps+1))
    arr[:,0] = 0
    arr[:,-1] = 1
    arr[:, 1:-1] = temp
    return np.diff(np.sort(arr))

if __name__ == '__main__':
    filepath = r'C:\\Users\\markz\\OneDrive\\Documents\\Datasets\\'
    #datafilename = filepath +'C1_Monahans_T90F_P1000_2000_LHS10000_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + '.csv'
    # Store suffix for dataset location. To store multiple components
    suffix = 'Monahans_T90F_P1000_2000_LHS10000_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + '.csv'

    T = 549.67 # R, 90 F

    # Monahans Clearfork Oil, Table 6.3 of Okuno
    z = np.array([0.0001, 0.3056, 0.2027, 0.1589, 0.2327, 0.1000])
    z_gas = np.array([0.95, 0.05, 0.0000, 0.0000, 0.0000, 0.0000])
    w = np.array([0.225, 0.008, 0.127, 0.240, 0.609, 1.042])
    Pc = np.array([1069.87, 667.20, 658.59, 487.51, 329.42, 258.78])  # [psia]
    Tc = np.array([547.56, 343.08, 612.02, 835.06, 1086.35, 1444.93])  # [R]
    BIP = np.array([[0.000, 0.094, 0.094, 0.094, 0.095, 0.095],
                    [0.094, 0.000, 0.000, 0.000, 0.000, 0.000],
                    [0.094, 0.000, 0.000, 0.000, 0.000, 0.000],
                    [0.094, 0.000, 0.000, 0.000, 0.000, 0.000],
                    [0.095, 0.000, 0.000, 0.000, 0.000, 0.000],
                    [0.095, 0.000, 0.000, 0.000, 0.000, 0.000]])

    NRtol = 1E-12
    NRmaxit = 100 # I think 10 is enough
    SStol = 1E-10  #
    tolSSSA = 1E-10
    SSmaxit = 500  # 1000000 # 1E6 might crash my computer.
    TolRR = 1E-10
    TolXz = 1E-8
    itSSSAmax = 1E6

    Nc = len(z)

    phase_num = 1
    row_index = 0

    # Instantiate class
    pr = pr()

    # Gen for range of P, T, x
    P_min = 1000 # [psia]
    P_max = 2000
    num_P = 100

    T_min = T
    T_max = T
    num_T = 1

    num_x = 10000

    for itx,P in enumerate(np.linspace(P_min, P_max, num_P)):
        Pr = P / Pc
        print('{} %'.format(itx/num_P*100))
        for T in np.linspace(T_min, T_max, num_T):
            Tr = T / Tc

            # Get all ai, bi values
            a_i, b_i = pr.aibi(P, T, w, Pr, Tr, Pc, Tc)

            # Get Vw mixing, part with BIPs and square roots
            Am = pr.Vw(Nc, a_i, BIP)

            # Gen all compositions by LHS
            XX = lhs_simplex(len(z), num_x)

            # Iterate the LHS comps

            for x in XX:
                sum_xiAij = pr.sum_a_interations(Nc, x, Am)

                a_mix = pr.am(x, sum_xiAij)
                b_mix = pr.bm(x, b_i)

                # All EOS variables defined, solve EOS
                Z = pr.Z_roots_calc(a_mix, b_mix)

                # If more than one root, skip
                if len(Z) > 1 and min(Z) > 0:
                    #print('2 positive real roots at P = {} bar, T = {} K.'.format(P,T))
                    continue
                else:
                    Z = max(Z)
                    ln_phi_x = pr.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)

                    # Store data for components
                    for i in range(0,6):
                        data_writer(filepath +'C'+str(i)+'_'+suffix, a_mix, b_mix, b_i[i], sum_xiAij[i], ln_phi_x[i])

                    #data_writer(datafilename, a_mix, b_mix, b_i[1], sum_xiAij[1], ln_phi_x[1])
