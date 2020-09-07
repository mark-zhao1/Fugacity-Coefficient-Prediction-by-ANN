'''
Inherit from pr from Stability_and_flash.py in Applied Model

Objective: Compare G_R_EOS and G_R_ANN. At const T, P. Vary z.

'''

from Stability_and_flash import pr

import math
import numpy as np
from cmath import pi
import cProfile, pstats, io

# If use model
import pandas as pd
import pickle
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib import rc


class GR(pr):
    def GR_plot(self, z, b_i, Am, Nc):
        # Modified to output Gr vs x_1 in a binary mixture, at const P, T.
        # Get parameters for Peng-Robinson EOS which are composition dependent.
        G_R_EOS = []
        G_R_ANN = []

        x_1 = []
        for z[0] in np.linspace(0,1,100):
            if z[0] == 0:
                continue
            z[1] = 1 - z[0]
            sum_xiAij = self.sum_a_interations(Nc, z, Am)
            a_mix = self.am(z, sum_xiAij)
            b_mix = self.bm(z, b_i)

            # All EOS variables defined, solve EOS
            Z = self.Z_roots_calc(a_mix, b_mix)

            # If not single real positive root, get ln_phi and root which correspond to lowest Gibbs energy
            if len(Z) > 1 and min(Z) > 0:
                print('Not single real positive root at x_1 = {}, Z = {}'.format(z[0], Z))
                ln_phi_z, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, z)
                d = ln_phi_z + np.log(z)
                G_R.append(np.dot(d, z))
                x_1.append(z[0])
            else:
                # EOS
                ln_phi_z_EOS = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)
                d = ln_phi_z_EOS + np.log(z)
                G_R_EOS.append(np.dot(d, z))
                # ANN
                ln_phi_z_ANN = self.ln_phi_model_calc(a_mix, b_mix, b_i, sum_xiAij)
                d = ln_phi_z_ANN + np.log(z)
                G_R_ANN.append(np.dot(d, z))

                x_1.append(z[0])

        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        plt.rc('text', usetex=True)

        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.plot(x_1, G_R_EOS, label='EOS, P = {} bar, T = {} K'.format(P,T))
        ax.plot(x_1, G_R_ANN, label='ANN, P = {} bar, T = {} K'.format(P,T))
        ax.set_ylabel(r'$\underline{G}_R$')
        ax.set_xlabel(r'$x_{C4}$')
        ax.set_title(r'$C_4 - C_{10}$ Binary Mixture')
        ax.legend()

        ax2 = fig.add_subplot(212)
        G_R_ANN = np.array(G_R_ANN)
        G_R_EOS = np.array(G_R_EOS)
        ax2.plot(x_1, G_R_ANN - G_R_EOS, label=r'Difference in $\underline{G}_{R}$')
        ax2.set_xlabel(r'$x_{C4}$')
        ax2.set_ylabel(r'$\underline{G}_{R ANN} - \underline{G}_{R EOS}$')
        ax2.legend()
        avg = np.average(abs(G_R_ANN - G_R_EOS))
        print(avg)
        plt.show()
        return

if __name__ =='__main__':
    ########################################################################################
    # INPUTS
    T = 620  # [K] [620, 650]
    P = 20.6  # [bar] [10, 30]

    # nC4-C10
    z = np.array([0.1, 0.9])
    w = np.array([0.193, 0.49])
    Pc = np.array([37.997, 21.1])  # [bar]
    Tc = np.array([425.2, 617.6])  # [K]
    BIP = np.zeros([2, 2])

    NRtol = 1E-12
    NRmaxit = 100  # I think 10 is enough
    SStol = 1E-10  #
    tolSSSA = 1E-10
    SSmaxit = 500  # 1000000 # 1E6 might crash my computer.
    TolRR = 1E-10
    TolXz = 1E-8
    itSSSAmax = 1E6

    # More global constants
    Tr = T / Tc
    Pr = P / Pc

    Nc = len(z)

    phase_num = 1
    row_index = 0

    #####################################################################################
    # Instantiate class
    pr = GR()

    # Use Model?
    pr.useModel = True

    # Load models
    modelPath = [
        r'C:\Users\win7\Desktop\logs\logs\scalars\lnphi_-10-10__100_4_20_100_20200825-164016',
        r'C:\Users\win7\Desktop\logs\logs\scalars\lnphi_nC10_T620-650_P10-30_100_4_20_100_20200826-174143'
    ]
    pipelinePath = [
        r'C:\Users\win7\Desktop\logs\logs\scalars\lnphi_-10-10__100_4_20_100_20200825-164016\full_pipeline_lnphi_-10-10__512_4_20_300_.pkl',
        r'C:\Users\win7\Desktop\logs\logs\scalars\lnphi_nC10_T620-650_P10-30_100_4_20_100_20200826-174143\full_pipeline_lnphi_nC10_T620-650_P10-30_100_4_20_100_.pkl'
    ]
    pr.load_ANN_model(modelPath, pipelinePath)

    # Parameters independent of composition placed out of loop.
    # Used in either stability analysis or 2-phase PT flash.

    # Get all K-values from Wilson
    K = pr.wilson_corr(Pr, Tr, w)
    ln_K = np.log(K)

    # Get all ai, bi values
    a_i, b_i = pr.aibi(P, T, w, Pr, Tr, Pc, Tc)

    # Get Vw mixing, part with BIPs and square roots
    Am = pr.Vw(Nc, a_i, BIP)
    ##########################################################################################
    pr.GR_plot(z, b_i, Am, Nc)

    print('END')