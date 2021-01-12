'''
Mod from data_gen.py

Component-specific data sets, do one at a time.
Components C3 and nC9
Generate training instances given a range of P, T, x.
Discard multiple positive real roots.

For expanded composition space, used:
T = [300, 650] K, 1000
P = [5, 400] bar, 1000
x_C3 = [0, 1], 100
100 million iterations, but discards instances with multiple roots.

'''

from Stability_and_flash import pr
import numpy as np
import csv
import os
import datetime

# Writes data
def data_writer(datafilename, a_mix, b_mix, b_i, sum_xiAij, ln_phi_x):
    b_i_max = np.max(b_i)
    templist = [a_mix, b_mix, b_i, sum_xiAij, ln_phi_x]

    # datafilename is defined at initiation
    with open(datafilename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(templist)
    return

if __name__ == '__main__':
    filepath = r'E:\\Datasets\\'
    datafilename = filepath +'nC9_data_C3-nC9_T300-650_P5-100_x0-1_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + '.csv'

    '''# INPUTS
    T = 650  # [K]
    P = 14.6    # [bar]
    '''

    # nC4-C10
    w = np.array([0.152, 0.44])
    z = np.array([0.8, 0.2])
    Pc = np.array([42.46, 23.1])  # [bar]
    Tc = np.array([369.8, 594.6])  # [K]
    BIP = np.zeros([2, 2])

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
    P_min = 5 # [bar]
    P_max = 100
    num_P = 320

    T_min = 300
    T_max = 650
    num_T = 1000

    x_nC4_min = 0
    x_nC4_max = 1
    num_x_nC4 = 100

    x = np.empty(2)

    for itx,P in enumerate(np.linspace(P_min, P_max, num_P)):
        Pr = P / Pc
        print('{} %'.format(itx/num_P*100))
        for T in np.linspace(T_min, T_max, num_T):
            Tr = T / Tc

            # Get all ai, bi values
            a_i, b_i = pr.aibi(P, T, w, Pr, Tr, Pc, Tc)

            # Get Vw mixing, part with BIPs and square roots
            Am = pr.Vw(Nc, a_i, BIP)

            for x[0] in np.linspace(x_nC4_min, x_nC4_max, num_x_nC4):
                x[1] = 1 - x[0]
                sum_xiAij = pr.sum_a_interations(Nc, x, Am)

                a_mix = pr.am(x, sum_xiAij)
                b_mix = pr.bm(x, b_i)

                # All EOS variables defined, solve EOS
                Z = pr.Z_roots_calc(a_mix, b_mix)

                #todo get this to reject multiple positive real roots.

                # If more than one root, get ln_phi and root which correspond to lowest Gibbs energy
                if len(Z) > 1 and min(Z) > 0:
                    #print('2 positive real roots at P = {} bar, T = {} K.'.format(P,T))
                    continue
                else:
                    Z = max(Z)
                    ln_phi_x = pr.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)
                    data_writer(datafilename, a_mix, b_mix, b_i[1], sum_xiAij[1], ln_phi_x[1])