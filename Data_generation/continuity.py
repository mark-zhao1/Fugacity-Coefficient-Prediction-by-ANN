from Stability_and_flash import pr
import numpy as np
import csv
import os
import datetime
import matplotlib.pyplot as plt

# Writes data
def data_writer(datafilename, a_mix, b_mix, b_i, sum_xiAij, ln_phi_x):
    b_i_max = np.max(b_i)
    templist = [a_mix, b_mix, b_i, sum_xiAij, ln_phi_x[0]]

    # datafilename is defined at initiation
    with open(datafilename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(templist)
    return

if __name__ == '__main__':

    filepath = r'E:\\Datasets\\'
    datafilename = filepath +'data_const_T_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + '.csv'

    # Some constants
    NRtol = 1E-12
    NRmaxit = 100
    SStol = 1E-10  #
    tolSSSA = 1E-10
    SSmaxit = 500  # 1000000
    TolRR = 1E-10
    TolXz = 1E-8
    itSSSAmax = 1E6

    R = 8.31446261815324
    sqrt2 = 1.41421356237309504
    twosqrt2 = 2 * sqrt2
    onepsqrt2 = 2.41421356237309504
    onemsqrt2 = -0.41421356237309504
    phase_num = 1
    row_index = 0

    pr = pr()

    # Gen for range of am, bm, b_i, sum
    a_mix_min = 0.001
    a_mix_max = 10.
    num_a_mix = 100

    b_mix_min = 0.246725#0.001
    b_mix_max = 0.246725#0.7
    num_b_mix = 1

    b_i_min = 0.134818#0.001
    b_i_max = 0.134818#0.7
    num_b_i = 1

    sum_x_min = 0.736707#0.001
    sum_x_max = 0.736707#10.
    num_sum_x = 1

    X = []
    y = []

    for a_mix in np.linspace(a_mix_min, a_mix_max, num_a_mix):
        for b_mix in np.linspace(b_mix_min, b_mix_max, num_b_mix):
            # Check if single root
            Z = pr.Z_roots_calc(a_mix, b_mix)
            if len(Z) > 1:
                continue
            else:
                for b_i in np.linspace(b_i_min, b_i_max, num_b_i):
                    for sum_xjAij in np.linspace(sum_x_min, sum_x_max, num_sum_x):
                        # Get lnphi
                        lnphi = pr.ln_phi_calc(b_i, a_mix, b_mix, sum_xjAij, Z)
                        # Store
                        X.append(a_mix)
                        y.append(lnphi)

plt.plot(X , y, 'ro')
plt.xlabel('a_mix')
plt.ylabel('lnphi')
plt.title('Lnphi vs a_mix, all else constant')
plt.show()

print(X)
print(y)