'''
Inherit from pr from Stability_and_flash.py

Objective: Compare G_R_EOS and G_R_ANN. At const T, P. Vary z in [0,1]

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

        # For TPD vs x, need to compute ln_phi_z and d_z
        sum_xiAij = self.sum_a_interations(Nc, z, Am)
        a_mix = self.am(z, sum_xiAij)
        b_mix = self.bm(z, b_i)
        Z = self.Z_roots_calc(a_mix, b_mix)
        ln_phi_z, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, z)
        d_z = ln_phi_z + np.log(z)

        G_R_EOS = []
        G_R_ANN = []
        TPD_EOS = []
        TPD_ANN = []

        x_1 = []
        x = np.empty(len(z))
        for x[0] in np.linspace(0,1,100):
            if x[0] == 0:
                continue
            x[1] = 1 - x[0]
            sum_xiAij = self.sum_a_interations(Nc, x, Am)
            a_mix = self.am(x, sum_xiAij)
            b_mix = self.bm(x, b_i)

            # All EOS variables defined, solve EOS
            Z = self.Z_roots_calc(a_mix, b_mix)

            # If not single real positive root, get ln_phi and root which correspond to lowest Gibbs energy
            if len(Z) > 1 and min(Z) > 0:
                print('Not single real positive root at x_1 = {}, Z = {}'.format(x[0], Z))
                ln_phi_x, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, x)
                d = ln_phi_x + np.log(x)
                G_R_EOS.append(np.dot(d, x))
                TPD_EOS.append(np.dot(x, d - d_z))

                G_R_ANN.append(np.nan)
                TPD_ANN.append(np.nan)

                x_1.append(x[0])
            else:
                # EOS
                ln_phi_x_EOS = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)
                d = ln_phi_x_EOS + np.log(x)
                G_R_EOS.append(np.dot(d, x))
                TPD_EOS.append(np.dot(x, d - d_z))
                # ANN
                ln_phi_x_ANN = self.ln_phi_model_calc(a_mix, b_mix, b_i, sum_xiAij)
                d = ln_phi_x_ANN + np.log(x)
                G_R_ANN.append(np.dot(d, x))
                TPD_ANN.append(np.dot(x, d - d_z))

                x_1.append(x[0])


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
        #plt.show()

        ###########################################################################

        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.plot(x_1, TPD_EOS, label='EOS, P = {} bar, T = {} K'.format(P, T))
        ax.plot(x_1, TPD_ANN, label='ANN, P = {} bar, T = {} K'.format(P, T))
        ax.set_ylabel(r'TPD')
        ax.set_xlabel(r'$x_{C4}$')
        ax.set_title(r'$C_4 - C_{10}$ TPD for z = '+str(z))
        ax.legend()

        ax2 = fig.add_subplot(212)
        TPD_ANN = np.array(TPD_ANN)
        TPD_EOS = np.array(TPD_EOS)
        ax2.plot(x_1, TPD_ANN - TPD_EOS, label=r'Difference in TPD')
        ax2.set_xlabel(r'$x_{C4}$')
        ax2.set_ylabel(r'$TPD_{ANN} - TPD_{EOS}$')
        ax2.legend()
        avg = np.average(abs(TPD_ANN - TPD_EOS))
        print(avg)
        plt.show()
        return

if __name__ =='__main__':
    ########################################################################################
    # Instantiate class
    pr = GR()

    # INPUTS
    T = 400  # [K] [620, 650]
    P = 45  # [bar] [10, 30]

    # nC4-C10
    z = np.array([0.65, 0.35])
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


    # Use Model?
    pr.useModel = True

    # Load models
    modelPath = [
        r'C:\Users\win7\Desktop\logs\logs\scalars\lnphi_nC4_T300-600_P5-100__100_4_20_100_20200916-112226',
        r'C:\Users\win7\Desktop\logs\logs\scalars\lnphi_nC10_T300-600_P5-100__100_4_20_100_20200915-230243'
    ]
    pipelinePath = [
        r'C:\Users\win7\Desktop\logs\logs\scalars\lnphi_nC4_T300-600_P5-100__100_4_20_100_20200916-112226\full_pipeline_lnphi_nC4_T300-600_P5-100__100_4_20_100_.pkl',
        r'C:\Users\win7\Desktop\logs\logs\scalars\lnphi_nC10_T300-600_P5-100__100_4_20_100_20200915-230243\full_pipeline_lnphi_nC4_T300-600_P5-100__100_4_20_100_.pkl'
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