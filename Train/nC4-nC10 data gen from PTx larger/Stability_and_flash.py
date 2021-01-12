'''
Created on Feb 24, 2020
Stability test and 2-phase flash

Inputs are around l.430

@author: markz
'''


import math
import numpy as np
from math import cos
from cmath import pi
import cProfile, pstats, io


class pr:
    def __init__(self):
        self.R = 8.31446261815324
        self.sqrt2 = 1.41421356237309504
        self.twosqrt2 = 2 * self.sqrt2
        self.onepsqrt2 = 2.41421356237309504
        self.onemsqrt2 = -0.41421356237309504

    ##########################################################################
    # Function definitions

    # Function bypassing problem with cubic root of small negative numbers
    def root3(self, num):
        if num < 0:
            return -(-num) ** (1. / 3.)
        else:
            return num ** (1. / 3.)


    # Returns K-values for all components using Wilson's correlation. Length of Pc, Tc, w must be equal.
    # K = y/x, x is liquid phase, y is vapor phase. Nth phase is liquid.

    def wilson_corr(self, Pr, Tr, w):
        #K = [1 / Pr[i] * math.exp(5.37 * (1 + w[i]) * (1 - 1 / Tr[i])) for i in range(len(w))]  # Truncated 5.373
        K = 1 / Pr * np.exp(5.37 * (1 + w) * (1 - 1 / Tr)) # Truncated 5.373
        # K = [1/Pr[i] * math.exp(5.373 * (1 + w[i]) * (1 - 1/Tr[i])) for i in range(len(w))]
        return K


    # Rachford-Rice
    # Get the phase compositions x, y from known K-values

    def objective_rr(self, beta, K, z):
        rrsum = ((K - 1) * z) / (1 + (K - 1) * beta)
        return np.sum(rrsum)

    # Analytical derivative of RR
    def rr_prime(self, beta, K, z):
        primesum = -1 * (z * (K - 1) ** 2) / (1 + beta * (K - 1)) ** 2
        return np.sum(primesum)


    # Solve RR equation for the vapor mole fraction beta
    #@profile
    def nr_beta(self, tol, K, beta):
        Kmin = np.min(K)
        Kmax = np.max(K)

        # Michelsen's window
        xl = 1 / (1 - Kmax)
        xr = 1 / (1 - Kmin)
        xg = beta

        # Find root NR method
        check = 1
        i = 0
        while i < NRmaxit:
            i += 1
            y = np.sum(((K - 1) * z) / (1 + (K - 1) * xg))
            fp = np.sum(-1 * (z * (K - 1) ** 2) / (1 + xg * (K - 1)) ** 2) # This is the gradient at xg
            xn = xg - y / fp
            if xn < xl:
                xn = 0.5 * (xg + xl)
            if xn > xr:
                xn = 0.5 * (xg + xr)

            if xg != 0:
                check = abs(xn - xg)
                if check < tol:
                    break
                else:
                    xg = xn
            else:
                xg = xn

        if i > NRmaxit:
            print('Trouble in NR Solve')
            print('it = {}'.format(i))
            print('beta = {}'.format(xg))
            print('K: {}'.format(K))
            print('z: {}'.format(z))
        else:
             return xn, i


    # Analytical 3rd order polynomial solver. Modified to output sorted real roots only.
    def cubic_real_roots(self, p):
        # Input p = [A, B, C] such that x**3 + A*x**2 + B*x + C = 0

        q = (p[0]**2 - 3 * p[1]) / 9
        r = (2 * p[0]**3 - 9 * p[0] * p[1] + 27 * p[2]) / 54
        qcub = q**3
        d = qcub - r**2

        if abs(qcub) < 1E-16 and abs(d) < 1E-16:
            # 3 repeated real roots. Same as single root.
            #nroot=1
            z = np.array([-p[0] / 3])
            return z
        if abs(d) < 1E-16 or (d > 0 and abs(d) > 1E-16):
            # 3 distinct real roots
            #nroot = 3
            th = math.acos(r/math.sqrt(qcub))
            sqQ = math.sqrt(q)
            z = np.empty(3)
            z[0] = -2 * sqQ * math.cos(th/3) - p[0] / 3
            z[1] = -2 * sqQ * math.cos((th+2*pi)/3) - p[0] / 3
            z[2] = -2 * sqQ * math.cos((th+4*pi)/3) - p[0] / 3
            return z
        else:
            # 1 real root, 2 complex conjugates
            #nroots = 1
            e = self.root3(math.sqrt(-d) + abs(r))
            if r > 0:
                e = -e
            z = np.array([e + q/e - p[0]/3])
            return z

    def Z_roots_calc(self, a_mix_phase, b_mix_phase):
        A = a_mix_phase # Optimized: Already has Pr, Tr. R is cancelled.
        B = b_mix_phase
        p = [-(1 - B), (A - 3 * B ** 2 - 2 * B), -(A * B - B ** 2 - B ** 3)]
        Z_roots = self.cubic_real_roots(p)
        return Z_roots

    def bm(self, phase_comps, b_i):
        return np.dot(phase_comps, b_i)

    def am(self, phase_comps, sum_xi_Aij):
        return np.dot(phase_comps, sum_xi_Aij)
    # Summation of a interactions, used in expression for lnphi

    def sum_a_interations(self, Nc, phase_comps, Am):
        sum_xi_Aij = np.zeros(Nc)
        '''
        for i in range(Nc):
            for j in range(Nc):
                sum_xi_Aij[i] += phase_comps[j] * Am[i, j]
        '''
        for i in range(Nc):
            sum_xi_Aij[i] = np.dot(phase_comps, Am[i, :])

        return sum_xi_Aij

    def ln_phi_calc(self, b_i, a_mix, b_mix, sum_xjAij, Z):
        # Get fugacity coeff for each component in each phase.

        a1 = b_i / b_mix * (Z - 1)
        a2 = - math.log(Z - b_mix)
        a3 = - 1 / (self.twosqrt2) * a_mix / b_mix
        a4a = sum_xjAij
        a4b = 2 / a_mix
        a4c = - b_i / b_mix
        a4 = a4a * a4b + a4c
        a5 = math.log((Z + self.onepsqrt2 * b_mix) / (Z + self.onemsqrt2 * b_mix))

        ln_phi = a1 + a2 + a3 * a4 * a5

        return ln_phi

    # Calculates mixing coefficients
    # Get a_mix_phase with mixing rule using a_i.
    def Vw(self, Nc,A,bip):
        Am = np.empty([Nc,Nc])
        for i in range(0,Nc):
            for j in range(0,Nc):
                Am[i,j] = np.sqrt(A[i] * A[j])*(1 - bip[i,j])
        return Am


    # Identify the phase stability result
    def caseid2(self, XX, itSSSAmax, TolXz, loop_count, sumXX):
        # Identify case
        tmp = abs(XX / z - 1)
        #tmp = [abs(XX[i] / z[i] - 1) for i in range(len(z))]
        if loop_count >= itSSSAmax:
            # Could not converge
            case_id = 1
        elif np.max(tmp) < TolXz:
            # Trivial case
            case_id = 2
        elif sumXX < 1:
            # Converged, but G of x higher than G of z
            case_id = 3
        else:
            # Two phase is more stable
            case_id = -1
        return case_id

    def two_phase_flash_iterate(self):
        K = self.wilson_corr(Pr, Tr, w) # If remove will have local variable clash with global
        beta = 0.5
        print(K)
        # Start looping here
        flag = 0
        outer_loop_count = 0

        ###################################
        # PROFILING
        # Single iteration of SS in two-phase flash.
        #for _ in range(100000):
        #    self.two_phase_flash_SS_test(Nc, K, flag, outer_loop_count, TolRR, b_i, Am, z)
        #return 'profiling', 'profiling'
        ###################################

        while outer_loop_count < SSmaxit and flag < 2:  # Flag exit condition at 2 to print converged+1 x, y, K-values
            print('SS Flash outer loop count: ' + str(outer_loop_count))
            outer_loop_count += 1
            # Call NR method for beta (vapor fraction)
            beta, i_count = self.nr_beta(TolRR, K, beta)

            print('Vapor frac: ' + str(beta))

            # Get Phase compositions from K and beta
            x = z / (1 + beta * (K - 1))
            y = K * x

            # Normalize
            x = x / np.sum(x)
            y = y / np.sum(y)

            # Check material balance for each component
            for comp in range(len(z)):
                if abs(z[comp] - (x[comp] * (1 - beta) + y[comp] * beta)) > 1E-10:
                    print('Caution: Material balance problem for component ' + str(comp))

            # Check mole fractions
            if 1 - np.sum(x) > 1E-12 or 1 - np.sum(y) > 1E-12:
                print('''Caution: Phase comp don't add up to 1.''')

            print('Liquid comp: ' + str(x))
            print('Vapor comp: ' + str(y))

            #####################################################
            # Liquid
            # Get parameters for Peng-Robinson EOS which are composition dependent.
            #a_mix, b_mix = ambm(x, b_i, Am)
            sum_xiAij = self.sum_a_interations(Nc, x, Am)
            a_mix = self.am(x, sum_xiAij)
            b_mix = self.bm(x, b_i)

            # All EOS variables defined, solve EOS for each phase
            Z = self.Z_roots_calc(a_mix, b_mix)

            if len(Z) > 1:
                print('SA: More than 1 root. Gibb\'s minimization performed.')
                ln_phi_x, Z = self.checkG(Nc, b_i, a_mix, b_mix, Am, Z, x)
            else:
                ln_phi_x = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)
            ######################################################
            # Vapor
            #a_mix, b_mix = ambm(y, b_i, Am)
            sum_xiAij = self.sum_a_interations(Nc, y, Am)
            a_mix = self.am(y, sum_xiAij)
            b_mix = self.bm(y, b_i)

            # All EOS variables defined, solve EOS for each phase
            Z = self.Z_roots_calc(a_mix, b_mix)

            if len(Z) > 1:
                print('SA: More than 1 root. Gibb\'s minimization performed.')
                ln_phi_y, Z = self.checkG(Nc, b_i, a_mix, b_mix, Am, Z, y)
            else:
                ln_phi_y = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)

            # Converge check
            ln_phi_diff = ln_phi_x - ln_phi_y
            c = np.abs(ln_phi_diff - np.log(K))
            if np.max(c) < SStol:
                flag += 1
                print('Exit flag:' + str(flag))
            else:
                flag = 0

            # Update K
            print('K old: ' + str(K))
            K = np.exp(ln_phi_diff)
            print('K new: ' + str(K))

            # For some reason, update x, y, ln_phi even after converged
            print('########################################')
        print('END 2-phase flash')
        return x, y

    def kappa(self, w):
        kappa = []  # Verified
        for comp in range(len(w)):
            if w[comp] <= 0.49:
                kappa.append(0.37464 + 1.54226 * w[comp] - 0.26992 * w[comp] ** 2)
            else:
                kappa.append(0.37964 + w[comp] * (1.48503 + w[comp] * (-0.164423 + w[comp] * 0.016666)))
        return np.array(kappa)

    def aibi(self, P,T,w,Pr,Tr,Pc,Tc):
        PT2 = P / T ** 2
        #PT = P / T
        Kappa = self.kappa(w)
        alpha = (1 + Kappa * (1 - np.sqrt(Tr))) ** 2
        a_i = 0.457236 * alpha * Tc ** 2 * PT2 / Pc
        b_i = 0.0778 * Pr / Tr  # Optimized Bi. Tr, Pr, removed R
        return a_i, b_i

    # Outputs lower Gibbs root and corresponding ln_phi
    def checkG(self, b_i, a_mix, b_mix, sum_xiAij, Z, x):
        Zmin = min(Z)
        Zmax = max(Z)
        if Zmin < 0:
            ln_phi_max = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Zmax)
            return ln_phi_max, Zmax

        ln_phi_min = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Zmin)
        ln_phi_max = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Zmax)

        arr = x * (ln_phi_min - ln_phi_max)
        if np.sum(arr) > 0:
            return ln_phi_max, Zmax
        else:
            return ln_phi_min, Zmin

    # For profiling
    def do_cprofile(self, func):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            repeats = int(1E6)
            try:
                for _ in range(repeats):
                    profile.enable()
                    result = func(*args, **kwargs)
                    profile.disable()
                return result
            finally:
                s = io.StringIO()
                sortby = 'tottime'
                ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
                ps.print_stats()
                print(s.getvalue())
                print('Profiled  %d repeats. Divide by that number for per iteration times.' % (repeats))
        return profiled_func

    # SA SS single iteration standalone, for profiling only.
    # Constant variables
    #@profile
    def SA_SS_single_it(self, Nc, x, b_i, Am, XX, d, tolSSSA, exit_flag):

        sum_xiAij = self.sum_a_interations(Nc, x, Am)
        b_mix = self.bm(x, b_i)
        a_mix = self.am(x, sum_xiAij)

        Z = self.Z_roots_calc(a_mix, b_mix)

        if len(Z) > 1:
            print('SA: More than 1 root. Gibb\'s minimization performed.')
            ln_phi_x, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, x)
        else:
            ln_phi_x = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)

        # Compute convergence
        tmp = np.abs(ln_phi_x + np.log(XX) - d)

        # Update XX
        XX = np.exp(d - ln_phi_x)

        # Update x
        sumXX = np.sum(XX)
        x = XX / sumXX

        # Check convergence
        if np.max(tmp) < tolSSSA:
            exit_flag += 1
        if exit_flag > 1:
            loop_count = it
            #break
        return

    # Two-phase flash SS single iteration standalone, for profiling only.
    #@profile
    def two_phase_flash_SS_test(self, Nc, K, flag, outer_loop_count, TolRR, b_i, Am, z):
        beta = 0.5
        while outer_loop_count < SSmaxit and flag < 2:  # Flag exit condition at 2 to print converged+1 x, y, K-values
            outer_loop_count += 1

            # Call NR method for beta (vapor fraction)
            beta, i_count = self.nr_beta(TolRR, K, beta)

            # Get Phase compositions from K and beta
            x = z / (1 + beta * (K - 1))
            y = K * x

            # Normalize
            x = x / np.sum(x)
            y = y / np.sum(y)

            #####################################################
            # Liquid
            # Get parameters for Peng-Robinson EOS which are composition dependent.
            #a_mix, b_mix = ambm(x, b_i, Am)
            sum_xiAij = self.sum_a_interations(Nc, x, Am)
            a_mix = self.am(x, sum_xiAij)
            b_mix = self.bm(x, b_i)

            # All EOS variables defined, solve EOS for each phase
            Z = self.Z_roots_calc(a_mix, b_mix)

            if len(Z) > 1:
                ln_phi_x, Z = self.checkG(Nc, b_i, a_mix, b_mix, Am, Z, x)
            else:
                ln_phi_x = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)
            ######################################################
            # Vapor
            #a_mix, b_mix = ambm(y, b_i, Am)
            sum_xiAij = self.sum_a_interations(Nc, y, Am)
            a_mix = self.am(y, sum_xiAij)
            b_mix = self.bm(y, b_i)

            # All EOS variables defined, solve EOS for each phase
            Z = self.Z_roots_calc(a_mix, b_mix)

            if len(Z) > 1:
                ln_phi_y, Z = self.checkG(Nc, b_i, a_mix, b_mix, Am, Z, y)
            else:
                ln_phi_y = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)

            # Converge check
            ln_phi_diff = ln_phi_x - ln_phi_y
            c = np.abs(ln_phi_diff - np.log(K))
            if np.max(c) < SStol:
                flag += 1
            else:
                flag = 0

            # Update K
            K = np.exp(ln_phi_diff)
        return

    #@do_cprofile
    def stability_analysis(self, T, P, z, b_i, Am, tolSSSA, itSSSAmax, Nc):
        # Get parameters for Peng-Robinson EOS which are composition dependent.
        #a_mix, b_mix = ambm(z, b_i, Am)
        sum_xiAij = self.sum_a_interations(Nc, z, Am)
        a_mix = self.am(z, sum_xiAij)
        b_mix = self.bm(z, b_i)

        # All EOS variables defined, solve EOS
        Z = self.Z_roots_calc(a_mix, b_mix)

        # If more than one root, get ln_phi and root which correspond to lowest Gibbs energy
        if len(Z) > 1:
            print('SA: More than 1 root. Gibb\'s minimization performed.')
            ln_phi_z, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, z)
        else:
            ln_phi_z = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)

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

            Z = self.Z_roots_calc(a_mix, b_mix)

            if len(Z) > 1:
                print('SA: More than 1 root. Gibb\'s minimization performed.')
                ln_phi_x, Z = self.checkG(b_i, a_mix, b_mix, sum_xiAij, Z, x)
            else:
                ln_phi_x = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)

            # Compute convergence
            tmp = np.abs(ln_phi_x + np.log(XX) - d)

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
        liq_case = self.caseid2(XX, itSSSAmax, TolXz, loop_count, sumXX)
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

            Z = self.Z_roots_calc(a_mix, b_mix)

            # Min G for roots, if more than 1 root.
            if len(Z) > 1:
                print('SA: More than 1 root. Gibb\'s minimization performed.')
                ln_phi_x, Z = self.checkG(b_i, a_mix, b_mix, Am, Z, x)
            else:
                ln_phi_x = self.ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)

            # Compute convergence
            tmp = np.abs(ln_phi_x + np.log(XX) - d)

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
        vap_case = self.caseid2(XX, itSSSAmax, TolXz, loop_count, sumXX)

        return sumXX_list, liq_case, vap_case

if __name__ == "__main__":
    ########################################################################################
    # INPUTS
    T = 374.85    # [K] Table 4-2 Ashutosh, Jaubert 2002 Fluid 1
    P = 97       # [bar]

    w = np.array([0.04, 0.225, 0.008, 0.098, 0.152, 0.1891, 0.2418, 0.296, 0.1725, 0.2778, 0.4076, 0.6234])
    z = np.array([0.0018, 0.0082, 0.2292, 0.0721, 0.0737, 0.0681, 0.0585, 0.0484, 0.1871, 0.1207, 0.0831, 0.0491])
    Pc = np.array([33.94, 73.76, 46.00, 48.84, 42.46, 37.65, 33.78, 29.69, 24.76, 19.31, 15.49, 11.98]) #[bar]
    Tc = np.array([126.20, 304.20, 190.60, 305.40, 369.80, 421.23, 466.06, 507.40, 661.61, 781.28, 912.58, 1145.28]) #[K]

    BIP = np.array([[0.0000, 0.0000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1300, 0.1300, 0.1300, 0.1300],
           [0.0000, 0.0000, 0.1000, 0.1450, 0.1873, 0.1645, 0.1414, 0.1286, 0.0532, 0.1003, 0.1244, 0.1368],
           [0.1000, 0.1000, 0.0000, 0.0420, 0.0420, 0.0420, 0.0420, 0.0420, 0.0527, 0.0581, 0.0653, 0.0806],
           [0.1000, 0.1450, 0.0420, 0.0000, 0.0400, 0.0400, 0.0400, 0.0400, 0.0417, 0.0424, 0.0433, 0.0451],
           [0.1000, 0.1873, 0.0420, 0.0400, 0.0000, 0.0300, 0.0300, 0.0300, 0.0334, 0.0378, 0.0468, 0.0661],
           [0.1000, 0.1645, 0.0420, 0.0400, 0.0300, 0.0000, 0.0116, 0.0155, 0.0452, 0.0594, 0.0693, 0.0788],
           [0.1000, 0.1414, 0.0420, 0.0400, 0.0300, 0.0116, 0.0000, 0.0058, 0.0317, 0.0521, 0.0698, 0.0898],
           [0.1000, 0.1286, 0.0420, 0.0400, 0.0300, 0.0155, 0.0058, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
           [0.1300, 0.0532, 0.0527, 0.0417, 0.0334, 0.0452, 0.0317, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
           [0.1300, 0.1003, 0.0581, 0.0424, 0.0378, 0.0594, 0.0521, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
           [0.1300, 0.1244, 0.0653, 0.0433, 0.0468, 0.0693, 0.0698, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
           [0.1300, 0.1368, 0.0806, 0.0451, 0.0661, 0.0788, 0.0898, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])


    NRtol = 1E-12
    NRmaxit = 100 # I think 10 is enough
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
    pr = pr()

    # Parameters independent of composition placed out of loop.
    # Used in either stability analysis or 2-phase PT flash.

    # Get all K-values from Wilson
    K = pr.wilson_corr(Pr, Tr, w)
    ln_K = np.log(K)

    # Get all ai, bi values
    a_i, b_i = pr.aibi(P, T, w, Pr, Tr, Pc, Tc)

    # Get Vw mixing, part with BIPs and square roots
    Am = pr.Vw(Nc,a_i,BIP)
    ##########################################################################################
    # Stability Analysis
    # Calculate constants ln_phi(z) ln(z[i])

    sumXX_list, liq_case, vap_case = pr.stability_analysis(T, P, z, b_i, Am, tolSSSA, itSSSAmax, Nc)

    # Get TPD
    TPD = -math.log(max(sumXX_list))

    print('At P = %s bar, and T = %s K' % (P, T))
    if liq_case < 0 or vap_case < 0:
        print('Single phase unstable, TPD = %s' % TPD)
        print('Run 2-phase flash.')

        phase_num = 2
        # Now call 2-phase flash func. Return only converged composition. Optimize by re-using calculated
        # variables.

        liq_comp, vap_comp = pr.two_phase_flash_iterate()
        print('liq and vap comp:')
        print(liq_comp, vap_comp)


    elif liq_case > 0 and vap_case > 0:
        print('Single phase stable')
        print('P = %s bar, T = %s K' % (P, T))
        print('Liq case: %d, Vap case: %d' % (liq_case, vap_case))
        # Copy single phase composition

    print('END')

    ''''''
    # Debug check if converged mole fractions are the same

    assert np.allclose(liq_comp, np.array([0.0017943234066841915, 0.008194852482489298, 0.22885842653547445, 0.07207896835248084, 0.07370807312111415,
     0.06812042987125054, 0.05852600921306141, 0.04842819785588235, 0.1872219235389395, 0.12078041822522835, 0.08315554889164768,
     0.04913282850574723]), rtol=1E-14, atol=0), 'Liq comp not exactly the same'

    assert np.allclose(vap_comp, np.array([0.010290202018209054, 0.0158988893032792, 0.7400746666429897, 0.10355600297209222, 0.061625428974152054,
     0.03754404425799916, 0.01959929654269055, 0.006225850892860039, 0.004744946752054136, 0.00042240741058238344, 1.8249073251744196e-05,
     1.515983949972891e-08]), rtol=1E-13, atol=0), 'Vap comp not exactly the same'

    liq_assert = np.isclose(liq_comp, np.array([0.0017943234066841915, 0.008194852482489298, 0.22885842653547445, 0.07207896835248084, 0.07370807312111415,
     0.06812042987125054, 0.05852600921306141, 0.04842819785588235, 0.1872219235389395, 0.12078041822522835, 0.08315554889164768,
     0.04913282850574723]), rtol=1E-13, atol=0)
    liq_incorrect_comps = np.argwhere(liq_assert)

    vap_assert = np.isclose(vap_comp, np.array([0.010290202018209054, 0.0158988893032792, 0.7400746666429897, 0.10355600297209222, 0.061625428974152054,
     0.03754404425799916, 0.01959929654269055, 0.006225850892860039, 0.004744946752054136, 0.00042240741058238344, 1.8249073251744196e-05,
     1.515983949972891e-08]), rtol=1E-13, atol=0)
    vap_incorrect_comps = np.argwhere(vap_assert)

    ''''''