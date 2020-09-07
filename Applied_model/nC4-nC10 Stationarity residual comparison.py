'''
Inherit from Stability_and_flash.py in Applied model

Objective: Compare EOS and ANN (Plot tmp stationarity residual vs iteration for ANN and EOS).

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

class newclass(pr):
    # Override class to plot tmp comparison
    def stability_analysis(self, T, P, z, b_i, Am, tolSSSA, itSSSAmax, Nc, K, TolXz):
        # Get parameters for Peng-Robinson EOS which are composition dependent.
        #a_mix, b_mix = ambm(z, b_i, Am)
        sum_xiAij = self.sum_a_interations(Nc, z, Am)
        a_mix = self.am(z, sum_xiAij)
        b_mix = self.bm(z, b_i)

        # if False forces EOS calc for z. Done only once for SA.
        # self.useModel
        if self.useModel:
            Z = self.Z_roots_det(a_mix, b_mix) # If multiple roots, returns array of roots. Else, returns False.
        else:
            Z = self.Z_roots_calc(a_mix, b_mix)

        if Z:
            # Use EOS lnphi
            if len(Z) > 1 and min(Z) > 0:
                print('SA: More than 1 root. Gibb\'s minimization performed.')
                ln_phi_z, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, z)
            else:
                ln_phi_z = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, max(Z))
        else:
            # Use ANN lnphi
            ln_phi_z = self.ln_phi_model_calc(a_mix, b_mix, b_i, sum_xiAij)

        d = ln_phi_z + np.log(z)
        #################
        # Liquid-like search for instability
        XX = z / K
        x = XX / np.sum(XX) # Maybe define sumXX beforehand
        # SS in SA
        exit_flag = 0
        ###############
        # PROFILING
        # For profiling. Single iteration of SS in SA.
        #for _ in range(1000000):
        #    self.SA_SS_single_it(Nc, x, b_i, Am, XX, d, tolSSSA, exit_flag)
        ###############

        for loop_count in range(int(itSSSAmax+1)):
            #a_mix, b_mix = ambm(x, b_i, Am)
            sum_xiAij = self.sum_a_interations(Nc, x, Am)
            a_mix = self.am(x, sum_xiAij)
            b_mix = self.bm(x, b_i)

            if self.useModel:
                Z = self.Z_roots_det(a_mix, b_mix)  # If multiple roots, returns array of roots. Else, returns False.
            else:
                Z = self.Z_roots_calc(a_mix, b_mix)

            if Z:
                # Use EOS lnphi
                if len(Z) > 1 and min(Z) > 0:
                    print('SA: More than 1 root. Gibb\'s minimization performed.')
                    ln_phi_x, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, x)
                else:
                    ln_phi_x = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, max(Z))
            else:
                # Use ANN lnphi
                ln_phi_x = self.ln_phi_model_calc(a_mix, b_mix, b_i, sum_xiAij)

            # Compute convergence by checking stationarity
            tmp = np.abs(ln_phi_x + np.log(XX) - d)
            # Log tmp (debug)
            if self.useModel:
                self.liq_tmp_ANN.append(tmp)
                self.liq_it_ANN.append(loop_count)
            else:
                self.liq_tmp_EOS.append(tmp)
                self.liq_it_EOS.append(loop_count)

            # Update XX
            XX = np.exp(d - ln_phi_x)

            # Update x
            sumXX = np.sum(XX)
            x = XX / sumXX

            # Check convergence
            if np.max(tmp) < tolSSSA:
                exit_flag += 1
            if exit_flag > 1:
                break

        sumXX_list = np.empty(2)
        sumXX_list[0] = sumXX
        liq_case = self.caseid2(XX, itSSSAmax, TolXz, loop_count, sumXX, z)
        #print('liq loop_count: {}'.format(loop_count))
        #################
        # Vapor-like search for instability
        XX = z * K
        x = XX / np.sum(XX)  # Maybe define sumXX beforehand

        # SS in SA
        exit_flag = 0

        for loop_count in range(int(itSSSAmax+1)):
            #a_mix, b_mix = ambm(x, b_i, Am)
            sum_xiAij = self.sum_a_interations(Nc, x, Am)
            a_mix = self.am(x, sum_xiAij)
            b_mix = self.bm(x, b_i)

            if self.useModel:
                Z = self.Z_roots_det(a_mix, b_mix)  # If multiple roots, returns array of roots. Else, returns False.
            else:
                Z = self.Z_roots_calc(a_mix, b_mix)

            if Z:
                # Use EOS lnphi
                if len(Z) > 1 and min(Z) > 0:
                    print('SA: More than 1 root. Gibb\'s minimization performed.')
                    ln_phi_x, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, x)
                else:
                    ln_phi_x = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, max(Z))
            else:
                # Use ANN lnphi
                ln_phi_x = self.ln_phi_model_calc(a_mix, b_mix, b_i, sum_xiAij)
            # Compute convergence
            tmp = np.abs(ln_phi_x + np.log(XX) - d)
            # Log tmp (debug)
            if self.useModel:
                self.vap_tmp_ANN.append(tmp)
                self.vap_it_ANN.append(loop_count)
            else:
                self.vap_tmp_EOS.append(tmp)
                self.vap_it_EOS.append(loop_count)

            # Update XX
            XX = np.exp(d - ln_phi_x)

            # Update x
            sumXX = np.sum(XX)
            x = XX / sumXX

            # Check convergence
            if np.max(tmp) < tolSSSA:
                exit_flag += 1
            if exit_flag > 1:
                break

        sumXX_list[1] = sumXX
        vap_case = self.caseid2(XX, itSSSAmax, TolXz, loop_count, sumXX, z)
        #print('vap loop_count: {}'.format(loop_count))

        return sumXX_list, liq_case, vap_case

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
    pr = newclass()

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
    # Define plotting vars
    pr.liq_tmp_ANN = []
    pr.liq_it_ANN = []
    pr.liq_tmp_EOS = []
    pr.liq_it_EOS = []

    pr.vap_tmp_ANN = []
    pr.vap_it_ANN = []
    pr.vap_tmp_EOS = []
    pr.vap_it_EOS = []

    for pr.useModel in [True, False]:
        sumXX_list, liq_case, vap_case = pr.stability_analysis(T, P, z, b_i, Am, tolSSSA, itSSSAmax, Nc, K, TolXz)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pr.vap_it_ANN, np.array(pr.vap_tmp_ANN)[:,0], label='ANN')
    ax.plot(pr.vap_it_EOS, np.array(pr.vap_tmp_EOS)[:,0], label='EOS')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$ln{x\phi_x}$ - $ln{z\phi_z}$')
    ax.set_title('SA Residual of Stationarity Equation vs Iteration')
    ax.legend()
    plt.show()
    print('END')