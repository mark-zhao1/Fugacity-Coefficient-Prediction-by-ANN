'''
Explore the domain of Case 1 Oil of Jose's manuscript.
Iterate through T, P
Do SA and 2-phase flash
Store: Am, Bm, Bi, sum_xjAij

Why? Use the information on the range of Am, Bm,... to decide training data domain.

NOTE: Some functions changed from Stability_and_flash.py
'''

import numpy as np
import math
import csv
import os
import datetime
#############################################
# Change Current working directory to script location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

##########################################################################
# Function definitions
# Writes data
def data_writer(a_mix, b_mix, b_i, sum_xiAij, ln_phi_x):
    b_i_max = np.max(b_i)
    templist = [a_mix, b_mix]
    templist.extend(b_i)
    templist.extend(sum_xiAij)
    templist.extend(ln_phi_x)

    # datafilename is defined at initiation
    with open(datafilename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(templist)
    return
# Function bypassing problem with cubic root of small negative numbers
def root3(num):
    if num < 0:
        return -(-num) ** (1. / 3.)
    else:
        return num ** (1. / 3.)


# Returns K-values for all components using Wilson's correlation. Length of Pc, Tc, w must be equal.
# K = y/x, x is liquid phase, y is vapor phase. Nth phase is liquid.

def wilson_corr(Pr, Tr, w):
    #K = [1 / Pr[i] * math.exp(5.37 * (1 + w[i]) * (1 - 1 / Tr[i])) for i in range(len(w))]  # Truncated 5.373
    K = 1 / Pr * np.exp(5.37 * (1 + w) * (1 - 1 / Tr)) # Truncated 5.373
    # K = [1/Pr[i] * math.exp(5.373 * (1 + w[i]) * (1 - 1/Tr[i])) for i in range(len(w))]
    return K


# Rachford-Rice
# Get the phase compositions x, y from known K-values

def objective_rr(beta, K, z):
    rrsum = ((K - 1) * z) / (1 + (K - 1) * beta)
    return np.sum(rrsum)

# Analytical derivative of RR
def rr_prime(beta, K, z):
    primesum = -1 * (z * (K - 1) ** 2) / (1 + beta * (K - 1)) ** 2
    return np.sum(primesum)


# Solve RR equation for the vapor mole fraction beta
#@profile
def nr_beta(tol, K, beta):
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
        fp = np.sum(-1 * (z * (K - 1) ** 2) / (1 + xg * (K - 1)) ** 2)
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
def cubic_real_roots(p):
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
        e = root3(math.sqrt(-d) + abs(r))
        if r > 0:
            e = -e
        z = np.array([e + q/e - p[0]/3])
        return z

def Z_roots_calc(a_mix_phase, b_mix_phase):
    A = a_mix_phase # Optimized: Already has Pr, Tr. R is cancelled.
    B = b_mix_phase
    p = [-(1 - B), (A - 3 * B ** 2 - 2 * B), -(A * B - B ** 2 - B ** 3)]
    Z_roots = cubic_real_roots(p)
    return Z_roots

def bm(phase_comps, b_i):
    return np.dot(phase_comps, b_i)

def am(phase_comps, sum_xi_Aij):
    return np.dot(phase_comps, sum_xi_Aij)
# Summation of a interactions, used in expression for lnphi

def sum_a_interations(Nc, phase_comps, Am):
    sum_xi_Aij = np.zeros(Nc)
    '''
    for i in range(Nc):
        for j in range(Nc):
            sum_xi_Aij[i] += phase_comps[j] * Am[i, j]
    '''
    for i in range(Nc):
        sum_xi_Aij[i] = np.dot(phase_comps, Am[i, :])

    return sum_xi_Aij

def ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z):
    # Get fugacity coeff for each component in each phase.

    a1 = b_i / b_mix * (Z - 1)
    a2 = - math.log(Z - b_mix)
    a3 = - 1 / (twosqrt2) * a_mix / b_mix
    a4a = sum_xiAij
    a4b = 2 / a_mix
    a4c = - b_i / b_mix
    a4 = a4a * a4b + a4c
    a5 = math.log((Z + onepsqrt2 * b_mix) / (Z + onemsqrt2 * b_mix))

    ln_phi = a1 + a2 + a3 * a4 * a5

    return ln_phi

# Calculates mixing coefficients
# Get a_mix_phase with mixing rule using a_i.
def Vw(Nc,A,bi):
    Am = np.empty([Nc,Nc])
    for i in range(0,Nc):
        for j in range(0,Nc):
            Am[i,j] = np.sqrt(A[i] * A[j])*(1 - bi[i,j])
    return Am


# Identify the phase stability result
def caseid2(XX, itSSSAmax, TolXz, loop_count, sumXX):
    # Identify case
    tmp = abs(XX / z - 1)
    #tmp = [abs(XX[i] / z[i] - 1) for i in range(len(z))]
    if loop_count > itSSSAmax:
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

def two_phase_flash_iterate():
    K = wilson_corr(Pr, Tr, w) # If remove will have local variable clash with global
    beta = 0.5
    print(K)
    # Start looping here
    flag = 0
    outer_loop_count = 0

    ###################################
    # PROFILING
    # Single iteration of SS in two-phase flash.
    #for _ in range(1000000):
    #    two_phase_flash_SS_single_it(Nc, K, flag, outer_loop_count, TolRR, b_i, Am, z)
    #return 'profiling', 'profiling'
    ###################################

    while outer_loop_count < SSmaxit and flag < 2:  # Flag exit condition at 2 to print converged+1 x, y, K-values
        print('SS Flash outer loop count: ' + str(outer_loop_count))
        outer_loop_count += 1
        # Call NR method for beta (vapor fraction)
        beta, i_count = nr_beta(TolRR, K, beta)

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
        sum_xiAij = sum_a_interations(Nc, x, Am)
        a_mix = am(x, sum_xiAij)
        b_mix = bm(x, b_i)

        # All EOS variables defined, solve EOS for each phase
        Z = Z_roots_calc(a_mix, b_mix)

        if len(Z) > 1:
            print('SA: More than 1 root. Gibb\'s minimization performed.')
            ln_phi_x, Z = checkG(Nc, b_i, a_mix, b_mix, Am, Z, x)
        else:
            ln_phi_x = ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)
            # Store a_mix, b_mix, b_i, sum_xiAij, ln_phi_x
            data_writer(a_mix, b_mix, b_i, sum_xiAij, ln_phi_x)

        ######################################################
        # Vapor
        #a_mix, b_mix = ambm(y, b_i, Am)
        sum_xiAij = sum_a_interations(Nc, y, Am)
        a_mix = am(y, sum_xiAij)
        b_mix = bm(y, b_i)

        # All EOS variables defined, solve EOS for each phase
        Z = Z_roots_calc(a_mix, b_mix)

        if len(Z) > 1:
            print('SA: More than 1 root. Gibb\'s minimization performed.')
            ln_phi_y, Z = checkG(Nc, b_i, a_mix, b_mix, Am, Z, y)
        else:
            ln_phi_y = ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)
            # Store a_mix, b_mix, b_i, sum_xiAij, ln_phi_x
            data_writer(a_mix, b_mix, b_i, sum_xiAij, ln_phi_y)

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

def kappa(w):
    kappa = []  # Verified
    for comp in range(len(w)):
        if w[comp] <= 0.49:
            kappa.append(0.37464 + 1.54226 * w[comp] - 0.26992 * w[comp] ** 2)
        else:
            kappa.append(0.37964 + w[comp] * (1.48503 + w[comp] * (-0.164423 + w[comp] * 0.016666)))
    return np.array(kappa)

def aibi(P,T,w,Pr,Tr,Pc,Tc):
    PT2 = P / T ** 2
    #PT = P / T
    Kappa = kappa(w)
    alpha = (1 + Kappa * (1 - np.sqrt(Tr))) ** 2
    a_i = 0.457236 * alpha * Tc ** 2 * PT2 / Pc
    b_i = 0.0778 * Pr / Tr  # Optimized Bi. Tr, Pr, removed R
    return a_i, b_i

# Outputs lower Gibbs root and corresponding ln_phi
def checkG(b_i, a_mix, b_mix, sum_xiAij, Z, x):
    Zmin = min(Z)
    Zmax = max(Z)
    if Zmin < 0:
        ln_phi_max = ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Zmax)
        return ln_phi_max, Zmax

    ln_phi_min = ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Zmin)
    ln_phi_max = ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Zmax)

    arr = x * (ln_phi_min - ln_phi_max)
    if np.sum(arr) > 0:
        return ln_phi_max, Zmax
    else:
        return ln_phi_min, Zmin


#@do_cprofile
def stability_analysis(T, P, z, b_i, Am, tolSSSA, itSSSAmax, Nc):
    # Get parameters for Peng-Robinson EOS which are composition dependent.
    #a_mix, b_mix = ambm(z, b_i, Am)
    sum_xiAij = sum_a_interations(Nc, z, Am)
    a_mix = am(z, sum_xiAij)
    b_mix = bm(z, b_i)

    # All EOS variables defined, solve EOS
    Z = Z_roots_calc(a_mix, b_mix)

    # If more than one root, get ln_phi and root which correspond to lowest Gibbs energy
    if len(Z) > 1:
        print('SA: More than 1 root. Gibb\'s minimization performed.')
        ln_phi_z, Z = checkG(b_i, a_mix, b_mix, sum_xiAij, Z, z)
    else:
        ln_phi_z = ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)

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
    #    SA_SS_single_it(Nc, x, b_i, Am, XX, d, tolSSSA, exit_flag)
    ###############
    for loop_count in range(int(itSSSAmax+1)):
        #a_mix, b_mix = ambm(x, b_i, Am)
        sum_xiAij = sum_a_interations(Nc, x, Am)
        a_mix = am(x, sum_xiAij)
        b_mix = bm(x, b_i)

        Z = Z_roots_calc(a_mix, b_mix)

        if len(Z) > 1:
            print('SA: More than 1 root. Gibb\'s minimization performed.')
            ln_phi_x, Z = checkG(b_i, a_mix, b_mix, sum_xiAij, Z, x)
        else:
            ln_phi_x = ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)
            # Store a_mix, b_mix, b_i, sum_xiAij, ln_phi_x
            data_writer(a_mix, b_mix, b_i, sum_xiAij, ln_phi_x)


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
    liq_case = caseid2(XX, itSSSAmax, TolXz, loop_count, sumXX)
    #################
    # Vapor-like search for instability
    XX = z * K
    x = XX / np.sum(XX)  # Maybe define sumXX beforehand

    # SS in SA
    exit_flag = 0
    for loop_count in range(int(itSSSAmax+1)):
        #a_mix, b_mix = ambm(x, b_i, Am)
        sum_xiAij = sum_a_interations(Nc, x, Am)
        a_mix = am(x, sum_xiAij)
        b_mix = bm(x, b_i)

        Z = Z_roots_calc(a_mix, b_mix)

        # Min G for roots, if more than 1 root.
        if len(Z) > 1:
            print('SA: More than 1 root. Gibb\'s minimization performed.')
            ln_phi_x, Z = checkG(b_i, a_mix, b_mix, Am, Z, x)
        else:
            ln_phi_x = ln_phi_calc(b_i, a_mix, b_mix, sum_xiAij, Z)
            # Store a_mix, b_mix, b_i, sum_xiAij, ln_phi_x
            data_writer(a_mix, b_mix, b_i, sum_xiAij, ln_phi_x)

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
    vap_case = caseid2(XX, itSSSAmax, TolXz, loop_count, sumXX)

    return sumXX_list, liq_case, vap_case


if __name__ =='__main__':
    ########################################################################################
    # INPUTS
    #T
    #P
    z = np.array([0.0861, 0.1503, 0.1671, 0.3304, 0.1611, 0.0713])
    z = z / (1 - 0.0337)    # Remove CO2 from z
    w = np.array([0.0080, 0.1305, 0.2404, 0.6177, 0.9566, 1.2683])
    Pc = np.array([667.1961, 652.5573, 493.0660, 315.4380, 239.8969, 238.1210])
    Pc = 0.0689475728 * Pc # Psia to [bar]
    Tc = np.array([288.0000, 619.5685, 833.7998, 1090.3544, 1351.8340, 1696.4620])
    Tc = 5/9 * Tc # R to [K]
    Nc = len(z)
    BIP = np.zeros((Nc,Nc))


    NRtol = 1E-12
    NRmaxit = 10  # 100 # I think 10 is enough
    SStol = 1E-10  #
    tolSSSA = 1E-10
    SSmaxit = 500  # 1000000 # 1E6 might crash my computer.
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
    datafilename = 'range_const_T_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) +'.csv'


    # Looping for T, P, x
    T_min = 650#250#250
    T_max = 651#750#800
    num_T = 1#100#100

    P_min = 15#10
    P_max = 90#90
    num_P = 10000#100

    # Iterate through T, P at constant z.

    for T in np.linspace(T_min, T_max, num_T):
        Tr = T / Tc
        for P in np.linspace(P_min, P_max, num_P):
            percent = (P - P_min)/(P_max - P_min)*100
            print('Progress: '+str(percent)+' %')

            Pr = P / Pc

            PT2 = P / T ** 2
            PT = P / T

            # Get all K-values from Wilson
            K = wilson_corr(Pr, Tr, w)  # Verified
            ln_K = np.log(K)

            # Get all ai, bi values
            a_i, b_i = aibi(P, T, w, Pr, Tr, Pc, Tc)

            # Get Vw mixing, part with BIPs and square roots
            Am = Vw(Nc, a_i, BIP)

            ## Stability analysis start here
            ## Write am, bm, b_i, sum_xiAij and ln_phi for single EOS root.
            sumXX_list, liq_case, vap_case = stability_analysis(T, P, z, b_i, Am, tolSSSA, itSSSAmax, Nc)


            print('At P = %s bar, and T = %s K' % (P, T))
            if liq_case < 0 or vap_case < 0:
                print('Single phase unstable')
                print('Run 2-phase flash.')

                phase_num = 2
                # Now call 2-phase flash func. Return only converged composition. Optimize by re-using calculated
                # variables.

                # Modified to write am, bm, b_i, sum_xiAij and ln_phi for single EOS root.
                liq_comp, vap_comp = two_phase_flash_iterate()
                print('liq and vap comp:')
                print(liq_comp, vap_comp)


            elif liq_case > 0 and vap_case > 0:
                print('Single phase stable')
                print('P = %s bar, T = %s K' % (P, T))
                print('Liq case: %d, Vap case: %d' % (liq_case, vap_case))
                # Copy single phase composition

            print('END')
