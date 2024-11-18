
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.linalg import sqrtm
import re

## size of the fock space
N = 3

## identity operators for a spin 1/2 and a coherent state
Id2 = qt.operators.identity(2)
IdN = qt.operators.identity(N)

#interfere early and late
a_1_2 = qt.tensor(qt.destroy(N), IdN)
a_2_2 = qt.tensor(IdN, qt.destroy(N))

# loss and reflection
a_1_3 = qt.tensor(qt.destroy(N), IdN, IdN)
a_2_3 = qt.tensor(IdN, qt.destroy(N), IdN)
a_3_3 = qt.tensor(IdN, IdN, qt.destroy(N))

## transmission and scattering beamsplitter
a_1_4 = qt.tensor(qt.destroy(N), IdN, IdN, IdN) 
a_4_4 = qt.tensor(IdN, IdN, IdN, qt.destroy(N))

## operators used for counting numbers 
Noperator = qt.num(N)

## Apd projection operators
Pj_01 = qt.composite(qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 1)))
# Pj_02 = qt.composite(qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 2)))
# Pj_03 = qt.composite(qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 3)))
#Pj_04 = qt.composite(qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 4)))

Pj_10 = qt.composite(qt.ket2dm(qt.basis(N, 1)), qt.ket2dm(qt.basis(N, 0)))
# Pj_20 = qt.composite(qt.ket2dm(qt.basis(N, 2)), qt.ket2dm(qt.basis(N, 0)))
# Pj_30 = qt.composite(qt.ket2dm(qt.basis(N, 3)), qt.ket2dm(qt.basis(N, 0)))
#Pj_40 = qt.composite(qt.ket2dm(qt.basis(N, 4)), qt.ket2dm(qt.basis(N, 0)))

Apd1 =  Pj_01 
Apd2 =  Pj_10

# Useful ideal state bases

psi_ideal_Xp = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
rho_ideal_Xp = qt.ket2dm(psi_ideal_Xp)
psi_ideal_Xm = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
rho_ideal_Xm = qt.ket2dm(psi_ideal_Xm)

psi_ideal_Yp = (qt.basis(2, 0) + 1j*qt.basis(2, 1)).unit()
rho_ideal_Yp = qt.ket2dm(psi_ideal_Yp)
psi_ideal_Ym = (qt.basis(2, 0) - 1j*qt.basis(2, 1)).unit()
rho_ideal_Ym = qt.ket2dm(psi_ideal_Ym)

psi_ideal_Zp = qt.basis(2, 0)
rho_ideal_Zp = qt.ket2dm(psi_ideal_Zp)
psi_ideal_Zm = qt.basis(2, 1)
rho_ideal_Zm = qt.ket2dm(psi_ideal_Zm) 

# Bell state |Φ+> = (|00> + |11>)/√2
phi_plus = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) +
            qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()

# Bell state |Φ-> = (|00> - |11>)/√2
phi_minus = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) -
             qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()

# Bell state |Ψ+> = (|01> + |10>)/√2
psi_plus = (qt.tensor(qt.basis(2, 0), qt.basis(2, 1)) +
            qt.tensor(qt.basis(2, 1), qt.basis(2, 0))).unit()

# Bell state |Ψ-> = (|01> - |10>)/√2
psi_minus = (qt.tensor(qt.basis(2, 0), qt.basis(2, 1)) -
             qt.tensor(qt.basis(2, 1), qt.basis(2, 0))).unit()

# Collect all Bell states in a list for convenience
bell_states = [phi_plus, phi_minus, psi_plus, psi_minus]

##################################################################
##################### MW and RF operations ###################

"""MW gates for one spin only/Change to make fidelity active."""
#define functions for setting up mw gates 

def set_mw_gates(fidelity, noise, gate_corrections):
    
    """pi half gate"""
    lam = 0
    phi = 0
    correction_pi2 = gate_corrections['pi_half']

    if fidelity == 'perfect':
        theta = np.pi/2
        g_pi2 = qt.Qobj([[np.cos(theta/2),  -np.exp(1j*lam)*np.sin(theta/2)],[np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]])
    elif fidelity == 'real':
        if noise == 0 : #stable
            theta = correction_pi2*np.pi/2
            g_pi2 = qt.Qobj([[np.cos(theta/2),  -np.exp(1j*lam)*np.sin(theta/2)],[np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]])
        elif noise == 1: #noisy
            theta = np.random.normal(loc=np.pi/2, scale=np.abs(correction_pi2-1)*np.pi/2)
            g_pi2 = qt.Qobj([[np.cos(theta/2),  -np.exp(1j*lam)*np.sin(theta/2)],[np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]])
    
    """pi gate"""
    lam = 0
    phi = 0
    correction_pi = gate_corrections['pi']
    if fidelity == 'perfect':
        theta = np.pi
        g_pi = qt.Qobj([[np.cos(theta/2),  -np.exp(1j*lam)*np.sin(theta/2)],[np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]])
    elif fidelity == 'real':
        if noise == 0:
            theta = correction_pi*np.pi
            g_pi = qt.Qobj([[np.cos(theta/2),  -np.exp(1j*lam)*np.sin(theta/2)],[np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]])
        elif noise == 1:
            theta = np.random.normal(loc=np.pi, scale=np.abs(correction_pi-1)*np.pi)
            g_pi = qt.Qobj([[np.cos(theta/2),  -np.exp(1j*lam)*np.sin(theta/2)],[np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]])
    
    dct = {'pi': g_pi,
            'pi_half': g_pi2
            }

    return dct

# calculate fidelities for angle corrections
def generate_mw_fid(corr_pi, corr_pi2):
    gate_corrections = {'pi': corr_pi,
                'pi_half': corr_pi2
                }
    pi = set_mw_gates('real', 0, gate_corrections)['pi']
    pi2 = set_mw_gates('real', 0, gate_corrections)['pi_half']
    pi_fid = qt.fidelity(pi*qt.basis(2,0), qt.basis(2,1))**2
    pi2_fid = qt.fidelity(pi2*qt.basis(2,0), (qt.basis(2,0)+ qt.basis(2,1)).unit())**2
    return pi_fid, pi2_fid

# make a list of corrections and corresponding fidelities
def generate_mw_fid_corr_dict(pi_list_corr, pi2_list_corr):

    pi_cor_fid = np.empty((0, 2), dtype=float)
    pi2_cor_fid = np.empty((0, 2), dtype=float)

    for i in range(len(pi_list_corr)):
        fids = generate_mw_fid(pi_list_corr[i], pi2_list_corr[i])
        pi_cor_fid = np.vstack((pi_cor_fid, np.array([[fids[0], pi_list_corr[i]]])))
        pi2_cor_fid = np.vstack((pi2_cor_fid, np.array([[fids[1], pi2_list_corr[i]]])))

    return pi_cor_fid, pi2_cor_fid

#given a fidelity you want for pi and pi2 find corrections to the angle of rotation in the list
def find_corr(fid_pi, fid_pi_half, array):
    
    #for pi
    index_pi = np.abs(array[0][:, 0] - fid_pi).argmin()
    corr_pi = array[0][index_pi, 1]

    #for pi half
    index_pi2 = np.abs(array[1][:, 0] - fid_pi_half).argmin()
    corr_pi2 = array[1][index_pi2, 1]

    # Print the corresponding second element
    return corr_pi, corr_pi2

# set mw pi and pi2 fidelity in one function, return the pi and pi2 operators
def set_mw_fidelities(fid = 'real', noise = 0, fidel_val = 1):
    fidpi = fidel_val['pi']
    fidpi2 = fidel_val['pi_half']
    
    #generate a list of corrections and fidelities for pi and pi half for the electron only 
    pi_list_corr = np.linspace(0.5, 1, 100)
    pi2_list_corr = np.linspace(0.2, 1, 100)
    result = generate_mw_fid_corr_dict(pi_list_corr, pi2_list_corr)

    corr = find_corr(fidpi, fidpi2, result)
    gate_corrections = {'pi': corr[0],
                'pi_half': corr[1]
                }
    gates = set_mw_gates(fid, noise, gate_corrections)
    return gates
    
"""Conditional gates with Si29."""

def cond_mw_gates(gate):
    
    pi_mw1 = qt.tensor(gate, qt.ket2dm(qt.basis(2, 0))) + qt.tensor(Id2, qt.ket2dm(qt.basis(2, 1)))
    pi_mw2 = qt.tensor(gate, qt.ket2dm(qt.basis(2, 1))) + qt.tensor(Id2, qt.ket2dm(qt.basis(2, 0)))

    cond_mw = {
        'pi_mw1': pi_mw1,
        'pi_mw2': pi_mw2
    }

    return cond_mw

##################################################################
##################### Beam splitter operations ###################

""" A general beam splitter operator for all linear photonic operators """
def general_BS(r, t, a1, a2):
        theta = np.arccos(abs(t))
        phase1 = np.angle(t) - np.angle(r)
        phase2 = np.angle(t) + np.angle(r)
        bs = ((-1j*phase1/2)*(a1.dag()*a1 - a2.dag()*a2)).expm()*\
                    (-theta*(a1.dag()*a2 - a2.dag()*a1)).expm()*\
                    ((-1j*phase2/2)*(a1.dag()*a1 - a2.dag()*a2)).expm()
        return bs

""" Electron Photon Entaglement beamsplitter """
def siv_beamsplitter(cav_refl, contrast):

    ## given complex reflection and transmission (in amplitude not intensity), and two lowering operators
    ## return the beam splitter operator
    
    #cavity outputs for A and B
    if contrast == 'real':
        r1_up = cav_refl['refl_refl']
        r1_down = cav_refl['nonrefl_refl']
        sc = cav_refl['refl_sc']
        transm = cav_refl['refl_tr']
        nsc = cav_refl['nonrefl_sc']
        ntransm = cav_refl['nonrefl_tr']
        
    elif contrast == 'perfect':
        r1_up = 1
        r1_down = 0
        sc = 0
        transm = 0
        nsc = 0
        ntransm = 1

    #commonly used beam splitter operations
    ## a_m_k destroy the mth mode out of k modes

    #interfere early and late
    a_1_2 = qt.tensor(qt.destroy(N), IdN)
    a_2_2 = qt.tensor(IdN, qt.destroy(N))

    # loss and reflection
    a_1_3 = qt.tensor(qt.destroy(N), IdN, IdN)
    a_2_3 = qt.tensor(IdN, qt.destroy(N), IdN)
    a_3_3 = qt.tensor(IdN, IdN, qt.destroy(N))

    ## transmission and scattering beamsplitter
    a_1_4 = qt.tensor(qt.destroy(N), IdN, IdN, IdN)
    a_4_4 = qt.tensor(IdN, IdN, IdN, qt.destroy(N))
    
    ## EARLY BIN INTERACTS WITH THE SIV
    theta1_up = np.arccos(np.sqrt(1 - abs(r1_up)**2))
    phase1_up_1 = 0 - np.angle(r1_up)
    phase1_up_2 = 0 + np.angle(r1_up)

    theta1_down = np.arccos(np.sqrt(1 - abs(r1_down)**2))
    phase1_down_1 = 0 - np.angle(r1_down)
    phase1_down_2 = 0 + np.angle(r1_down)

    # Early Reflection and loss channels
    bs1_up = (((-1j*phase1_up_1/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm()*\
                (-theta1_up*(a_1_3.dag()*a_3_3 - a_3_3.dag()*a_1_3)).expm()*\
                ((-1j*phase1_up_2/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm())

    bs1_down = (((-1j*phase1_down_1/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm()*\
                (-theta1_down*(a_1_3.dag()*a_3_3 - a_3_3.dag()*a_1_3)).expm()*\
                ((-1j*phase1_down_2/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm())

    bs2_up = general_BS(sc, transm, a_1_4, a_4_4)
    bs2_down = general_BS(nsc, ntransm, a_1_4, a_4_4)

    #operation to reflect early bin
    oper1 = qt.tensor(qt.ket2dm(qt.basis(2, 0)), bs1_up) + qt.tensor(qt.ket2dm(qt.basis(2, 1)), bs1_down)
    #operation to split transmission and scattering for early bin
    oper2 = qt.tensor(qt.ket2dm(qt.basis(2, 0)),bs2_up) + qt.tensor(qt.ket2dm(qt.basis(2, 1)),bs2_down)
    return oper1, oper2


""" Electron Photon Entaglement beamsplitter with Si29 """
def siv_beamsplitter_si29(cav_refl, contrast):

    ## given complex reflection and transmission (in amplitude not intensity), and two lowering operators
    ## return the beam splitter operator
    
    #cavity outputs for A and B
    if contrast == 'real':
        r1_up = cav_refl['refl_refl']
        r1_down = cav_refl['nonrefl_refl']
        sc = cav_refl['refl_sc']
        transm = cav_refl['refl_tr']
        nsc = cav_refl['nonrefl_sc']
        ntransm = cav_refl['nonrefl_tr']
        
    elif contrast == 'perfect':
        r1_up = 1
        r1_down = 0
        sc = 0
        transm = 0
        nsc = 0
        ntransm = 1

    #commonly used beam splitter operations
    ## a_m_k destroy the mth mode out of k modes

    #interfere early and late
    a_1_2 = qt.tensor(qt.destroy(N), IdN)
    a_2_2 = qt.tensor(IdN, qt.destroy(N))

    # loss and reflection
    a_1_3 = qt.tensor(qt.destroy(N), IdN, IdN)
    a_2_3 = qt.tensor(IdN, qt.destroy(N), IdN)
    a_3_3 = qt.tensor(IdN, IdN, qt.destroy(N))

    ## transmission and scattering beamsplitter
    a_1_4 = qt.tensor(qt.destroy(N), IdN, IdN, IdN)
    a_4_4 = qt.tensor(IdN, IdN, IdN, qt.destroy(N))
    
    ## EARLY BIN INTERACTS WITH THE SIV
    theta1_up = np.arccos(np.sqrt(1 - abs(r1_up)**2))
    phase1_up_1 = 0 - np.angle(r1_up)
    phase1_up_2 = 0 + np.angle(r1_up)

    theta1_down = np.arccos(np.sqrt(1 - abs(r1_down)**2))
    phase1_down_1 = 0 - np.angle(r1_down)
    phase1_down_2 = 0 + np.angle(r1_down)

    # Early Reflection and loss channels
    bs1_up = (((-1j*phase1_up_1/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm()*\
                (-theta1_up*(a_1_3.dag()*a_3_3 - a_3_3.dag()*a_1_3)).expm()*\
                ((-1j*phase1_up_2/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm())

    bs1_down = (((-1j*phase1_down_1/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm()*\
                (-theta1_down*(a_1_3.dag()*a_3_3 - a_3_3.dag()*a_1_3)).expm()*\
                ((-1j*phase1_down_2/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm())

    bs2_up = general_BS(sc, transm, a_1_4, a_4_4)
    bs2_down = general_BS(nsc, ntransm, a_1_4, a_4_4)

    #operation to reflect early bin
    oper1 = qt.tensor(qt.ket2dm(qt.basis(2, 0)), Id2, bs1_up) + qt.tensor(qt.ket2dm(qt.basis(2, 1)), Id2, bs1_down)
    #operation to split transmission and scattering for early bin
    oper2 = qt.tensor(qt.ket2dm(qt.basis(2, 0)), Id2, bs2_up) + qt.tensor(qt.ket2dm(qt.basis(2, 1)), Id2, bs2_down)

    return oper1, oper2

""" Electron Photon Entaglement beamsplitter of the first node for serial ee entanglement """
def siv_beamsplitter_ee_e1_serial(cav_refl, contrast):

    ## given complex reflection and transmission (in amplitude not intensity), and two lowering operators
    ## return the beam splitter operator
    
    #cavity outputs for A and B
    if contrast == 'real':
        r1_up = cav_refl['refl_refl']
        r1_down = cav_refl['nonrefl_refl']
        sc = cav_refl['refl_sc']
        transm = cav_refl['refl_tr']
        nsc = cav_refl['nonrefl_sc']
        ntransm = cav_refl['nonrefl_tr']
        
    elif contrast == 'perfect':
        r1_up = 1
        r1_down = 0
        sc = 0
        transm = 0
        nsc = 0
        ntransm = 1

    #commonly used beam splitter operations
    ## a_m_k destroy the mth mode out of k modes

    #interfere early and late
    a_1_2 = qt.tensor(qt.destroy(N), IdN)
    a_2_2 = qt.tensor(IdN, qt.destroy(N))

    # loss and reflection
    a_1_3 = qt.tensor(qt.destroy(N), IdN, IdN)
    a_2_3 = qt.tensor(IdN, qt.destroy(N), IdN)
    a_3_3 = qt.tensor(IdN, IdN, qt.destroy(N))

    ## transmission and scattering beamsplitter
    a_1_4 = qt.tensor(qt.destroy(N), IdN, IdN, IdN)
    a_4_4 = qt.tensor(IdN, IdN, IdN, qt.destroy(N))
    
    ## EARLY BIN INTERACTS WITH THE SIV
    theta1_up = np.arccos(np.sqrt(1 - abs(r1_up)**2))
    phase1_up_1 = 0 - np.angle(r1_up)
    phase1_up_2 = 0 + np.angle(r1_up)

    theta1_down = np.arccos(np.sqrt(1 - abs(r1_down)**2))
    phase1_down_1 = 0 - np.angle(r1_down)
    phase1_down_2 = 0 + np.angle(r1_down)

    # Early Reflection and loss channels
    bs1_up = (((-1j*phase1_up_1/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm()*\
                (-theta1_up*(a_1_3.dag()*a_3_3 - a_3_3.dag()*a_1_3)).expm()*\
                ((-1j*phase1_up_2/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm())

    bs1_down = (((-1j*phase1_down_1/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm()*\
                (-theta1_down*(a_1_3.dag()*a_3_3 - a_3_3.dag()*a_1_3)).expm()*\
                ((-1j*phase1_down_2/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm())

    bs2_up = general_BS(sc, transm, a_1_4, a_4_4)
    bs2_down = general_BS(nsc, ntransm, a_1_4, a_4_4)

    #operation to reflect early bin
    oper1 = qt.tensor(qt.ket2dm(qt.basis(2, 0)), Id2, bs1_up) + qt.tensor(qt.ket2dm(qt.basis(2, 1)), Id2, bs1_down)
    #operation to split transmission and scattering for early bin
    oper2 = qt.tensor(qt.ket2dm(qt.basis(2, 0)), Id2, bs2_up) + qt.tensor(qt.ket2dm(qt.basis(2, 1)), Id2, bs2_down)

    return oper1, oper2

""" Electron Photon Entaglement beamsplitter of the second node for serial ee entanglement """
def siv_beamsplitter_ee_e2_serial(cav_refl, contrast):

    ## given complex reflection and transmission (in amplitude not intensity), and two lowering operators
    ## return the beam splitter operator
    
    #cavity outputs for A and B
    if contrast == 'real':
        r1_up = cav_refl['refl_refl']
        r1_down = cav_refl['nonrefl_refl']
        sc = cav_refl['refl_sc']
        transm = cav_refl['refl_tr']
        nsc = cav_refl['nonrefl_sc']
        ntransm = cav_refl['nonrefl_tr']
        
    elif contrast == 'perfect':
        r1_up = 1
        r1_down = 0
        sc = 0
        transm = 0
        nsc = 0
        ntransm = 1

    #commonly used beam splitter operations
    ## a_m_k destroy the mth mode out of k modes

    #interfere early and late
    a_1_2 = qt.tensor(qt.destroy(N), IdN)
    a_2_2 = qt.tensor(IdN, qt.destroy(N))

    # loss and reflection
    a_1_3 = qt.tensor(qt.destroy(N), IdN, IdN)
    a_2_3 = qt.tensor(IdN, qt.destroy(N), IdN)
    a_3_3 = qt.tensor(IdN, IdN, qt.destroy(N))

    ## transmission and scattering beamsplitter
    a_1_4 = qt.tensor(qt.destroy(N), IdN, IdN, IdN)
    a_4_4 = qt.tensor(IdN, IdN, IdN, qt.destroy(N))
    
    ## EARLY BIN INTERACTS WITH THE SIV
    theta1_up = np.arccos(np.sqrt(1 - abs(r1_up)**2))
    phase1_up_1 = 0 - np.angle(r1_up)
    phase1_up_2 = 0 + np.angle(r1_up)

    theta1_down = np.arccos(np.sqrt(1 - abs(r1_down)**2))
    phase1_down_1 = 0 - np.angle(r1_down)
    phase1_down_2 = 0 + np.angle(r1_down)

    # Early Reflection and loss channels
    bs1_up = (((-1j*phase1_up_1/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm()*\
                (-theta1_up*(a_1_3.dag()*a_3_3 - a_3_3.dag()*a_1_3)).expm()*\
                ((-1j*phase1_up_2/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm())

    bs1_down = (((-1j*phase1_down_1/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm()*\
                (-theta1_down*(a_1_3.dag()*a_3_3 - a_3_3.dag()*a_1_3)).expm()*\
                ((-1j*phase1_down_2/2)*(a_1_3.dag()*a_1_3 - a_3_3.dag()*a_3_3)).expm())

    bs2_up = general_BS(sc, transm, a_1_4, a_4_4)
    bs2_down = general_BS(nsc, ntransm, a_1_4, a_4_4)

    #operation to reflect early bin
    oper1 = qt.tensor(Id2, qt.ket2dm(qt.basis(2, 0)), bs1_up) + qt.tensor(Id2, qt.ket2dm(qt.basis(2, 1)), bs1_down)
    #operation to split transmission and scattering for early bin
    oper2 = qt.tensor(Id2, qt.ket2dm(qt.basis(2, 0)), bs2_up) + qt.tensor(Id2, qt.ket2dm(qt.basis(2, 1)), bs2_down)

    return oper1, oper2
##################################################################
##################### Spin Photon gates ##########################

""" Electron Photon Entanglement: take a qubit in and entangle it with a photonic time-bin qubit """
def spin_photon_entaglement(SiV_beamsplitter, el_initial, pi, mu):
    
    alpha = np.sqrt(mu)
    early_time_bin = qt.tensor(qt.coherent(N, alpha/np.sqrt(2)), qt.coherent(N, 0))
    late_time_bin = qt.tensor(qt.coherent(N, 0), qt.coherent(N, alpha/np.sqrt(2)))
    input_coh = (early_time_bin + late_time_bin)
    rho_0 = qt.tensor(el_initial, qt.ket2dm(input_coh))

    # print('Initial number of photons per qubit =', (Noperator*rho_0.ptrace([1])).tr(),  (Noperator*rho_0.ptrace([2])).tr())

    # reflect early
    rho_1 = SiV_beamsplitter[0]*(qt.tensor(rho_0, qt.fock_dm(N, 0)))*SiV_beamsplitter[0].dag()
    rho_2 = (SiV_beamsplitter[1]*(qt.tensor(rho_1, qt.fock_dm(N, 0)))*SiV_beamsplitter[1].dag()).ptrace([0, 2, 3])

    # do a pi gate on the electron
    pi_oper = qt.tensor(pi, IdN, IdN)
    rho_3 = pi_oper*rho_2*pi_oper.dag()
    
    # print('The number of photons mid spin photon =', (Noperator*rho_2.ptrace([1])).tr(), (Noperator*rho_2.ptrace([2])).tr())

    # reflect late
    rho_4 = SiV_beamsplitter[0]*(qt.tensor(rho_3, qt.fock_dm(N, 0)))*SiV_beamsplitter[0].dag()
    rho_5 = (SiV_beamsplitter[1]*(qt.tensor(rho_4, qt.fock_dm(N, 0)))*SiV_beamsplitter[1].dag()).ptrace([0, 2, 3])

    return rho_5

""" Electron Photon Entanglement with si29 attached: take an el-nucleus in and entangle el with a photonic time-bin qubit """
def spin_photon_entanglement_withsi29(SiV_beamsplitter, eln_initial, pi, mu):
    
    alpha = np.sqrt(mu)
    early_time_bin = qt.tensor(qt.coherent(N, alpha/np.sqrt(2)), qt.coherent(N, 0))
    late_time_bin = qt.tensor(qt.coherent(N, 0), qt.coherent(N, alpha/np.sqrt(2)))
    input_coh = (early_time_bin + late_time_bin)
    rho_0 = qt.tensor(eln_initial, qt.ket2dm(input_coh))

    # print('Initial number of photons per qubit =', (Noperator*rho_0.ptrace([1])).tr(),  (Noperator*rho_0.ptrace([2])).tr())

    # reflect early
    rho_1 = SiV_beamsplitter[0]*(qt.tensor(rho_0, qt.fock_dm(N, 0)))*SiV_beamsplitter[0].dag()
    rho_2 = (SiV_beamsplitter[1]*(qt.tensor(rho_1, qt.fock_dm(N, 0)))*SiV_beamsplitter[1].dag()).ptrace([0,1, 3, 4])

    # do a pi gate on the electron
    pi_oper = qt.tensor(pi, Id2, IdN, IdN)
    rho_3 = pi_oper*rho_2*pi_oper.dag()
    
    # print('The number of photons mid spin photon =', (Noperator*rho_2.ptrace([1])).tr(), (Noperator*rho_2.ptrace([2])).tr())

    # reflect late
    rho_4 = SiV_beamsplitter[0]*(qt.tensor(rho_3, qt.fock_dm(N, 0)))*SiV_beamsplitter[0].dag()
    rho_5 = (SiV_beamsplitter[1]*(qt.tensor(rho_4, qt.fock_dm(N, 0)))*SiV_beamsplitter[1].dag()).ptrace([0,1, 3, 4])

    return rho_5

## PHONE gate
def nucleus_photon_entaglement(SiV_beamsplitter, el_initial, si29_initial, cond_pi, mu):

    ## state init
    alpha = np.sqrt(mu)
    
    # early_time_bin = qt.tensor(qt.coherent(N, alpha/np.sqrt(2)), qt.coherent(N, 0))
    # late_time_bin = qt.tensor(qt.coherent(N, 0), qt.coherent(N, alpha/np.sqrt(2)))
    # input_coh = (early_time_bin + late_time_bin)
    # # rho_0 = qt.tensor(el_initial, si29_initial, qt.ket2dm(input_coh))

    rho_0 = qt.tensor(el_initial, si29_initial, qt.ket2dm(qt.coherent(N, alpha/np.sqrt(2))), qt.ket2dm(qt.coherent(N, alpha/np.sqrt(2))))

    # print('Initial number of photons per qubit =', (Noperator*rho_0.ptrace([2])).tr(), (Noperator*rho_0.ptrace([3])).tr())

    # reflect early
    rho_1 = SiV_beamsplitter[0]*(qt.tensor(rho_0, qt.fock_dm(N, 0)))*SiV_beamsplitter[0].dag()
    rho_2 = (SiV_beamsplitter[1]*(qt.tensor(rho_1, qt.fock_dm(N, 0)))*SiV_beamsplitter[1].dag()).ptrace([0,1, 3, 4])

    # conditional mw1
    MW1_oper = qt.tensor(cond_pi, IdN, IdN)
    rho_3 = MW1_oper*rho_2*MW1_oper.dag()

    # print('Initial number of photons per qubit =', (Noperator*rho_2.ptrace([2])).tr(), (Noperator*rho_2.ptrace([3])).tr())

    # reflect late
    rho_4 = SiV_beamsplitter[0]*(qt.tensor(rho_3, qt.fock_dm(N, 0)))*SiV_beamsplitter[0].dag()
    rho_5 = (SiV_beamsplitter[1]*(qt.tensor(rho_4, qt.fock_dm(N, 0)))*SiV_beamsplitter[1].dag()).ptrace([0, 1, 3, 4])
    
    # conditional mw1
    rho_6 = MW1_oper*rho_5*MW1_oper.dag()

    # print('number of photons per qubit after the entanglement =', (Noperator*rho_5.ptrace([2])).tr(), (Noperator*rho_5.ptrace([3])).tr())

    return rho_6

##################################################################
##################### Measurements ###############################

## ELectron-Photon tomography

def el_photon_bell_state_Ztomography(rho):

    ## Only one apd fires - not the other
    Apd1_oper = qt.composite(Id2, Apd1)
    Apd2_oper = qt.composite(Id2, Apd2)

    rho_final_z_apd1 = ((Apd1_oper*rho*Apd1_oper.dag())/(Apd1_oper*rho*Apd1_oper.dag()).tr()).ptrace([0])
    rho_final_z_apd2 = ((Apd2_oper*rho*Apd2_oper.dag())/(Apd2_oper*rho*Apd2_oper.dag()).tr()).ptrace([0])

    #Zminus - APD2, Zplus - APD1
    ZpZp = qt.fidelity(rho_final_z_apd1, rho_ideal_Zp)**2
    ZpZm = qt.fidelity(rho_final_z_apd1, rho_ideal_Zm)**2
    ZmZp = qt.fidelity(rho_final_z_apd2, rho_ideal_Zp)**2
    ZmZm = qt.fidelity(rho_final_z_apd2, rho_ideal_Zm)**2
    
    ZZ = [ZpZp/(ZpZp+ZpZm+ZmZp+ZmZm), ZpZm/(ZpZp+ZpZm+ZmZp+ZmZm), ZmZp/(ZpZp+ZpZm+ZmZp+ZmZm), ZmZm/(ZpZp+ZpZm+ZmZp+ZmZm)]
    return ZZ

def el_photon_bell_state_Xtomography(rho):

    ## Only one apd fires - not the other
    Apd1_oper = qt.composite(Id2, Apd1)
    Apd2_oper = qt.composite(Id2, Apd2)

    #beam splitter for interfering early and late to change into X basis
    bs_el = general_BS(np.sqrt(1/2), np.sqrt(1/2), a_1_2, a_2_2)

    #operation of interfering early and late bins on a 50/50 BS
    oper_el = qt.tensor(Id2, bs_el)
    rho1 = oper_el*rho*oper_el.dag()

    rho_final_x_apd1 = ((Apd1_oper*rho1*Apd1_oper.dag())/(Apd1_oper*rho1*Apd1_oper.dag()).tr()).ptrace([0])
    rho_final_x_apd2 = ((Apd2_oper*rho1*Apd2_oper.dag())/(Apd2_oper*rho1*Apd2_oper.dag()).tr()).ptrace([0])
    
    #Xm - xminus - APD2, xplus - APD1
    XpXp = qt.fidelity(rho_final_x_apd1, rho_ideal_Xp)**2
    XpXm = qt.fidelity(rho_final_x_apd1, rho_ideal_Xm)**2
    XmXp = qt.fidelity(rho_final_x_apd2, rho_ideal_Xp)**2
    XmXm = qt.fidelity(rho_final_x_apd2, rho_ideal_Xm)**2

    XX = [XpXp/(XpXp+XpXm+XmXp+XmXm), XpXm/(XpXp+XpXm+XmXp+XmXm), XmXp/(XpXp+XpXm+XmXp+XmXm), XmXm/(XpXp+XpXm+XmXp+XmXm)]

    return XX

def el_photon_bell_state_Ytomography(rho):

    ## Only one apd fires - not the other
    Apd1_oper = qt.composite(Id2, Apd1)
    Apd2_oper = qt.composite(Id2, Apd2)

    bs_el_y = general_BS(1j*np.sqrt(1/2), np.sqrt(1/2), a_1_2, a_2_2)
    
    #operation of interfering early and late bins on a 50/50 BS
    oper_el_y = qt.tensor(Id2, bs_el_y)
    rho1 = oper_el_y*rho*oper_el_y.dag()

    rho_final_y_apd1 = ((Apd1_oper*rho1*Apd1_oper.dag())/(Apd1_oper*rho1*Apd1_oper.dag()).tr()).ptrace([0])
    rho_final_y_apd2 = ((Apd2_oper*rho1*Apd2_oper.dag())/(Apd2_oper*rho1*Apd2_oper.dag()).tr()).ptrace([0])
    
    #Yminus - APD2, Yplus - APD1
    YpYp = qt.fidelity(rho_final_y_apd1, rho_ideal_Yp)**2
    YpYm = qt.fidelity(rho_final_y_apd1, rho_ideal_Ym)**2
    YmYp = qt.fidelity(rho_final_y_apd2, rho_ideal_Yp)**2
    YmYm = qt.fidelity(rho_final_y_apd2, rho_ideal_Ym)**2
    
    YY = [YpYp/(YpYp+YpYm+YmYp+YmYm), YpYm/(YpYp+YpYm+YmYp+YmYm), YmYp/(YpYp+YpYm+YmYp+YmYm), YmYm/(YpYp+YpYm+YmYp+YmYm)]

    return YY

def fidelity_bases(tomography_output):
        fid_ZZ =  tomography_output['ZZ'][0] + tomography_output['ZZ'][3]
        fid_XX =  tomography_output['XX'][0] + tomography_output['XX'][3] - tomography_output['XX'][1] - tomography_output['XX'][2]
        fid_YY =  tomography_output['YY'][0] + tomography_output['YY'][3] - tomography_output['YY'][1] - tomography_output['YY'][2]
        fid_total = fid_ZZ/3 + fid_XX/3 + fid_YY/3

        fid = {
            'fid_ZZ': fid_ZZ,
            'fid_XX': fid_XX,
            'fid_YY': fid_YY,
            'fid_total': fid_total

        }
        return fid

## ELectron-electron tomography 

def elel_bell_state_Ztomography(rho):
    ZpZp = qt.fidelity(rho, qt.tensor(rho_ideal_Zp, rho_ideal_Zp))**2
    ZpZm = qt.fidelity(rho, qt.tensor(rho_ideal_Zp, rho_ideal_Zm))**2
    ZmZp = qt.fidelity(rho, qt.tensor(rho_ideal_Zm, rho_ideal_Zp))**2
    ZmZm = qt.fidelity(rho, qt.tensor(rho_ideal_Zm, rho_ideal_Zm))**2
    
    ZZ = [ZpZp/(ZpZp+ZpZm+ZmZp+ZmZm), ZpZm/(ZpZp+ZpZm+ZmZp+ZmZm), ZmZp/(ZpZp+ZpZm+ZmZp+ZmZm), ZmZm/(ZpZp+ZpZm+ZmZp+ZmZm)]
    return ZZ

def elel_bell_state_Xtomography(rho):
    XpXp = qt.fidelity(rho, qt.tensor(rho_ideal_Xp, rho_ideal_Xp))**2
    XpXm = qt.fidelity(rho, qt.tensor(rho_ideal_Xp, rho_ideal_Xm))**2
    XmXp = qt.fidelity(rho, qt.tensor(rho_ideal_Xm, rho_ideal_Xp))**2
    XmXm = qt.fidelity(rho, qt.tensor(rho_ideal_Xm, rho_ideal_Xm))**2
    XX = [XpXp/(XpXp+XpXm+XmXp+XmXm), XpXm/(XpXp+XpXm+XmXp+XmXm), XmXp/(XpXp+XpXm+XmXp+XmXm), XmXm/(XpXp+XpXm+XmXp+XmXm)]

    return XX

def elel_bell_state_Ytomography(rho):
    YpYp = qt.fidelity(rho, qt.tensor(rho_ideal_Yp, rho_ideal_Yp))**2
    YpYm = qt.fidelity(rho, qt.tensor(rho_ideal_Yp, rho_ideal_Ym))**2
    YmYp = qt.fidelity(rho, qt.tensor(rho_ideal_Ym, rho_ideal_Yp))**2
    YmYm = qt.fidelity(rho, qt.tensor(rho_ideal_Ym, rho_ideal_Ym))**2
    
    YY = [YpYp/(YpYp+YpYm+YmYp+YmYm), YpYm/(YpYp+YpYm+YmYp+YmYm), YmYp/(YpYp+YpYm+YmYp+YmYm), YmYm/(YpYp+YpYm+YmYp+YmYm)]
    return YY
#######################################################################################
##################### For telescope (maybe move it from here) #########################

"""LO-qubit interference"""
def interfere_qubit_with_LO(rho, mu_LO, phi_LO):

    counts = 0
    # add LO time-bins to the total state: si29 - early_qubit - late_qubit - early_LO - late_LO

    alpha_LO = np.sqrt(mu_LO)
    rho_0 = qt.tensor(rho, qt.ket2dm(qt.coherent(N, alpha_LO)), qt.ket2dm(qt.coherent(N, np.exp(1j*phi_LO)*alpha_LO)))

    # print('number of photons per bin at the start =', (Noperator*rho_0.ptrace([2])).tr(), (Noperator*rho_0.ptrace([3])).tr(), (Noperator*rho_0.ptrace([4])).tr(), (Noperator*rho_0.ptrace([5])).tr())

    # 50/50 BS operator for interferening LO/qubit early 
    ratio = 0.5
    bs_q_early = general_BS(np.sqrt(1 - ratio), 1 - np.sqrt(ratio), a_1_3, a_3_3)

    # interfere early time-bins
    oper_qubit_LO_early = qt.tensor(Id2, Id2, bs_q_early, IdN)
    rho_1 = oper_qubit_LO_early*rho_0*oper_qubit_LO_early.dag()

    # print('number of photons per bin after early intereference =', (Noperator*rho_1.ptrace([2])).tr(), (Noperator*rho_1.ptrace([3])).tr(),(Noperator*rho_1.ptrace([4])).tr(), (Noperator*rho_1.ptrace([5])).tr())

    # 50/50 BS operator for interferening LO/qubit late 
    ratio = 0.5
    bs_q_late = general_BS(np.sqrt(1 - ratio), 1 - np.sqrt(ratio), a_1_3, a_3_3)

    # interfere late time-bins
    oper_qubit_LO_late = qt.tensor(Id2, Id2, IdN, bs_q_late)
    rho_2 = oper_qubit_LO_late*rho_1*oper_qubit_LO_late.dag()

    # print('number of photons per bin after late intereference =', (Noperator*rho_2.ptrace([2])).tr(), (Noperator*rho_2.ptrace([3])).tr(),(Noperator*rho_2.ptrace([4])).tr(), (Noperator*rho_2.ptrace([5])).tr())

    counts = {
        'counts_early_1': np.abs((Noperator*rho_2.ptrace([2])).tr()),
        'counts_early_2': np.abs((Noperator*rho_2.ptrace([4])).tr()),
        'counts_late_1': np.abs((Noperator*rho_2.ptrace([3])).tr()),
        'counts_late_2': np.abs((Noperator*rho_2.ptrace([5])).tr()),
    }

    return counts, rho_2.ptrace([0, 1])

""" Add linear loss for a time-bin photonic qubit tensored with a two qubits el and el """
def loss_photonqubit_ee_serial(rho, eff):

    bs_e = general_BS(1j*np.sqrt(1 - eff), np.sqrt(eff), a_1_3, a_3_3)
    bs_l = general_BS(1j*np.sqrt(1 - eff), np.sqrt(eff), a_2_3, a_3_3)

    #operation of BS1 50/50 on early reflected beam
    oper_e = qt.tensor(Id2, Id2, bs_e)
    rho_1 = (oper_e*(qt.tensor(rho, qt.fock_dm(N, 0)))*oper_e.dag()).ptrace([0, 1, 2, 3])

    #operation of BS1 50/50 on late reflected beam
    oper_l = qt.tensor(Id2, Id2, bs_l)
    rho_2 = (oper_l*(qt.tensor(rho_1, qt.fock_dm(N, 0)))*oper_l.dag()).ptrace([0, 1, 2, 3])
    
    # print('The number of photons after loss =', (Noperator*rho_2.ptrace([1])).tr() + (Noperator*rho_2.ptrace([2])).tr())

    return rho_2

def phi_photon_measurement_ee_serial(rho, phi, tdi_noise = 0):

    # define a TDI
    ratio = np.random.normal(loc=0.5, scale=0*0.1*0.5)
    angle = np.random.normal(loc=2*np.pi + tdi_noise, scale=0*0.1*2*np.pi)
    r = np.exp(1j*(angle + phi))*np.sqrt(ratio)
    if np.abs(r) > 1:
        r = 1
    
    bs_5050_el_r = general_BS(r, np.sqrt(1-(abs(r))**2), a_1_2, a_2_2)
    
    #operation of interfering early and late bins on a 50/50 BS
    oper7 = qt.tensor(Id2, Id2, bs_5050_el_r)
    rho_13 = oper7*rho*oper7.dag()
    
    # measure (in blind experiment we selected for 1 photon events)
    Pj_01 = qt.composite(Id2, Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 1)))  # removed Id2
    # Pj_02 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 2)))
    # Pj_03 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 3)))
    
    Pj_10 = qt.composite(Id2, Id2, qt.ket2dm(qt.basis(N, 1)), qt.ket2dm(qt.basis(N, 0)))  # removed Id2
    # Pj_20 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 2)), qt.ket2dm(qt.basis(N, 0)))
    # Pj_30 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 3)), qt.ket2dm(qt.basis(N, 0)))
    
    #overall
    P_apd1 = Pj_01 #+ Pj_02 + Pj_03 # apd1 fires -> late time-bin
    P_apd2 = Pj_10 #+ Pj_20 + Pj_30 # apd2 fires -> early time-bin
    
    #Final density matrix of the electron-photon state
    rho_final_b_apd1 = ((P_apd1*rho_13*P_apd1.dag())/(P_apd1*rho_13*P_apd1.dag()).tr()).ptrace([0, 1]) # spin state left over after apd 1
    rho_final_b_apd2 = ((P_apd2*rho_13*P_apd2.dag())/(P_apd2*rho_13*P_apd2.dag()).tr()).ptrace([0, 1]) # spin state left over after apd2

    # probability of each apd firing
    brate_apd_1 = (P_apd1*rho_13*P_apd1.dag()).tr()
    brate_apd_2 = (P_apd2*rho_13*P_apd2.dag()).tr()
    bnorm_apd_rates = brate_apd_1 + brate_apd_2
    bprob_apd1 = brate_apd_1 / bnorm_apd_rates # probability of apd1 firing
    bprob_apd2 = brate_apd_2 / bnorm_apd_rates # probability of apd2 firing
    if np.abs(1 - (bprob_apd1 + bprob_apd2)) < 0.001: 
        pass   #this is in case trace above yields an approximation, in which case probs wont sum to 1 which yields error at choice
    else:
        return "Error: probabilities of apd1 and apd2 firing do not sum to 1"
    # probabilistic projective measurement
    quantum_measurement = np.random.choice([1,2], p=[bprob_apd1, bprob_apd2])
    if quantum_measurement == 1:
        spin_state = rho_final_b_apd1
    elif quantum_measurement == 2:
        spin_state = rho_final_b_apd2
    return spin_state, quantum_measurement-1, brate_apd_1, brate_apd_2, brate_apd_1 + brate_apd_2 # apd1 fires --> m = 1, apd2 fires --> m = 0

##################################################################
##################### Fidelity errors  #########################

# try this function for fidelity errors
def calculate_fidelity_uncertainty(rho, sigma, rho_errors):
    # Convert rho and sigma to Qobj if they are not already
    rho_qobj = qt.Qobj(rho)
    sigma_qobj = qt.Qobj(sigma)
    
    # Calculate fidelity
    F = (qt.fidelity(rho_qobj, sigma_qobj))**2
    
    # Calculate partial derivatives of fidelity with respect to each element of rho
    partial_F_rho = np.zeros(rho.shape, dtype=complex)
    
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            # Derivatives with respect to rho
            delta_rho = np.zeros(rho.shape, dtype=complex)
            delta_rho[i, j] = 1e-6
            
            # Ensure Hermiticity
            delta_rho[j, i] = np.conj(delta_rho[i, j])
            
            # Ensure trace preservation
            trace_adjustment = np.trace(delta_rho)
            delta_rho[i, i] -= trace_adjustment / rho.shape[0]
            
            # Calculate perturbed fidelity
            perturbed_rho = qt.Qobj(rho + delta_rho)
            perturbed_fidelity = qt.fidelity(perturbed_rho, sigma_qobj)**2
            partial_F_rho[i, j] = (perturbed_fidelity - F) / 1e-6
    
    # Calculate the uncertainty in fidelity
    uncertainty = np.sqrt(np.sum((np.abs(partial_F_rho) * rho_errors) ** 2))
    
    return F, uncertainty

##################################################################
##################### To retrieve density matrices from a csv ####

# Function to clean and convert string to NumPy array, handling complex numbers
def clean_and_convert_to_array(s):
    try:
        # Remove 'array(' and ')' from the string
        cleaned_str = s.strip().replace('array(', '').replace(')', '')
        
        # Replace any newlines or multiple spaces with a single space
        cleaned_str = re.sub(r'\s+', ' ', cleaned_str)

        # Add commas between complex numbers and elements
        # cleaned_str = re.sub(r'\]\s*\[', '],[', cleaned_str)

        # Final cleanup and conversion to a list
        cleaned_str = cleaned_str.replace('[ ', '[').replace(' ]', ']').replace(' ', ',')

        # Safely evaluate the string to convert it into a list of lists or array
        matrix = eval(cleaned_str)  # Use eval carefully with cleaned string

        return np.array(matrix)
    except Exception as e:
        print(f"Conversion failed for: {cleaned_str}, Error: {e}")
        return np.nan

##################################################################
##################### Photon measurement #########################

def phi_photon_measurement(rho, phi, tdi_noise = 0):

    # define a TDI
    ratio = np.random.normal(loc=0.5, scale=0*0.1*0.5)
    angle = np.random.normal(loc=2*np.pi + tdi_noise, scale=0*0.1*2*np.pi)
    r = np.exp(1j*(angle + phi))*np.sqrt(ratio)
    if np.abs(r) > 1:
        r = 1
    
    bs_5050_el_r = general_BS(r, np.sqrt(1-(abs(r))**2), a_1_2, a_2_2)
    
    #operation of interfering early and late bins on a 50/50 BS
    oper7 = qt.tensor(Id2, bs_5050_el_r)
    rho_13 = oper7*rho*oper7.dag()
    
    # measure (in blind experiment we selected for 1 photon events)
    Pj_01 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 1)))  # removed Id2
    # Pj_02 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 2)))
    # Pj_03 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 3)))
    
    Pj_10 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 1)), qt.ket2dm(qt.basis(N, 0)))  # removed Id2
    # Pj_20 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 2)), qt.ket2dm(qt.basis(N, 0)))
    # Pj_30 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 3)), qt.ket2dm(qt.basis(N, 0)))
    
    #overall
    P_apd1 = Pj_01 #+ Pj_02 + Pj_03 # apd1 fires -> late time-bin
    P_apd2 = Pj_10 #+ Pj_20 + Pj_30 # apd2 fires -> early time-bin
    
    #Final density matrix of the electron-photon state
    rho_final_b_apd1 = ((P_apd1*rho_13*P_apd1.dag())/(P_apd1*rho_13*P_apd1.dag()).tr()).ptrace([0]) # spin state left over after apd 1
    rho_final_b_apd2 = ((P_apd2*rho_13*P_apd2.dag())/(P_apd2*rho_13*P_apd2.dag()).tr()).ptrace([0]) # spin state left over after apd2

    # probability of each apd firing
    brate_apd_1 = (P_apd1*rho_13*P_apd1.dag()).tr()
    brate_apd_2 = (P_apd2*rho_13*P_apd2.dag()).tr()
    bnorm_apd_rates = brate_apd_1 + brate_apd_2
    bprob_apd1 = brate_apd_1 / bnorm_apd_rates # probability of apd1 firing
    bprob_apd2 = brate_apd_2 / bnorm_apd_rates # probability of apd2 firing
    if np.abs(1 - (bprob_apd1 + bprob_apd2)) < 0.001: 
        pass   #this is in case trace above yields an approximation, in which case probs wont sum to 1 which yields error at choice
    else:
        return "Error: probabilities of apd1 and apd2 firing do not sum to 1"
    # probabilistic projective measurement
    quantum_measurement = np.random.choice([1,2], p=[bprob_apd1, bprob_apd2])
    if quantum_measurement == 1:
        spin_state = rho_final_b_apd1
    elif quantum_measurement == 2:
        spin_state = rho_final_b_apd2
    return spin_state, quantum_measurement-1, brate_apd_1, brate_apd_2, brate_apd_1 + brate_apd_2 # apd1 fires --> m = 1, apd2 fires --> m = 0

""" Measure the photonic qubit qith el2 + n2 """
def phi_photon_measurement_withsi29(rho, phi, tdi_noise = 0):
    ratio = np.random.normal(loc=0.5, scale=0*0.1*0.5)
    angle = np.random.normal(loc=2*np.pi + tdi_noise, scale=0*0.1*2*np.pi)
    r = np.exp(1j*(angle + phi))*np.sqrt(ratio)
    if np.abs(r) > 1:
        r = 1
    
    bs_5050_el_r = general_BS(r, np.sqrt(1-(abs(r))**2), a_1_2, a_2_2)
    
    #operation of interfering early and late bins on a 50/50 BS
    oper7 = qt.tensor(Id2,Id2, bs_5050_el_r)
    rho_13 = oper7*rho*oper7.dag()
    
    # measure (in blind experiment we selected for 1 photon events)
    Pj_01 = qt.composite(Id2, Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 1)))  # removed Id2 
    Pj_10 = qt.composite(Id2, Id2, qt.ket2dm(qt.basis(N, 1)), qt.ket2dm(qt.basis(N, 0)))  # removed Id2

    #overall
    P_apd1 = Pj_01 #+ Pj_02 + Pj_03 # apd1 fires -> late time-bin
    P_apd2 = Pj_10 #+ Pj_20 + Pj_30 # apd2 fires -> early time-bin
    
    #Final density matrix of the electron-photon state
    rho_final_b_apd1 = ((P_apd1*rho_13*P_apd1.dag())/(P_apd1*rho_13*P_apd1.dag()).tr()).ptrace([0, 1]) # spin state left over after apd 1
    rho_final_b_apd2 = ((P_apd2*rho_13*P_apd2.dag())/(P_apd2*rho_13*P_apd2.dag()).tr()).ptrace([0, 1]) # spin state left over after apd2

    # probability of each apd firing
    brate_apd_1 = (P_apd1*rho_13*P_apd1.dag()).tr()
    brate_apd_2 = (P_apd2*rho_13*P_apd2.dag()).tr()
    bnorm_apd_rates = brate_apd_1 + brate_apd_2
    bprob_apd1 = brate_apd_1 / bnorm_apd_rates # probability of apd1 firing
    bprob_apd2 = brate_apd_2 / bnorm_apd_rates # probability of apd2 firing
    if np.abs(1 - (bprob_apd1 + bprob_apd2)) < 0.001: 
        pass   #this is in case trace above yields an approximation, in which case probs wont sum to 1 which yields error at choice
    else:
        return "Error: probabilities of apd1 and apd2 firing do not sum to 1"
    # probabilistic projective measurement
    quantum_measurement = np.random.choice([1,2], p=[bprob_apd1, bprob_apd2])
    if quantum_measurement == 1:
        spin_state = rho_final_b_apd1
    elif quantum_measurement == 2:
        spin_state = rho_final_b_apd2
    return spin_state, quantum_measurement-1, brate_apd_1, brate_apd_2, brate_apd_1 + brate_apd_2 # apd1 fires --> m = 1, apd2 fires --> m = 0

""" Add linear loss for a time-bin photonic qubit tensored with a two qubits el + Si29 """
def loss_photonqubit_elSpin_withsi29(rho, eff):

    bs_e = general_BS(1j*np.sqrt(1 - eff), np.sqrt(eff), a_1_3, a_3_3)
    bs_l = general_BS(1j*np.sqrt(1 - eff), np.sqrt(eff), a_2_3, a_3_3)

     #operation of BS1 50/50 on early reflected beam
    oper_e = qt.tensor(Id2, Id2, bs_e)
    rho_1 = (oper_e*(qt.tensor(rho, qt.fock_dm(N, 0)))*oper_e.dag()).ptrace([0, 1, 2, 3])

    #operation of BS1 50/50 on late reflected beam
    oper_l = qt.tensor(Id2, Id2, bs_l)
    rho_2 = (oper_l*(qt.tensor(rho_1, qt.fock_dm(N, 0)))*oper_l.dag()).ptrace([0, 1, 2, 3])
    
    # print('The number of photons after loss =', (Noperator*rho_2.ptrace([1])).tr() + (Noperator*rho_2.ptrace([2])).tr())

    return rho_2

""" Add linear loss for a time-bin photonic qubit tensored with a single qubit """
def loss_photonqubit_elSpin(rho, eff):
    bs_e = general_BS(1j*np.sqrt(1 - eff), np.sqrt(eff), a_1_3, a_3_3)
    bs_l = general_BS(1j*np.sqrt(1 - eff), np.sqrt(eff), a_2_3, a_3_3)

     #operation of BS1 50/50 on early reflected beam
    oper_e = qt.tensor(Id2, bs_e)
    rho_1 = (oper_e*(qt.tensor(rho, qt.fock_dm(N, 0)))*oper_e.dag()).ptrace([0, 1, 2])

    #operation of BS1 50/50 on late reflected beam
    oper_l = qt.tensor(Id2, bs_l)
    rho_2 = (oper_l*(qt.tensor(rho_1, qt.fock_dm(N, 0)))*oper_l.dag()).ptrace([0, 1, 2])
    
    # print('The number of photons after loss =', (Noperator*rho_2.ptrace([1])).tr() + (Noperator*rho_2.ptrace([2])).tr())

    return rho_2

""" Add linear loss for a time-bin photonic qudit tensored with a 3 qubits: el1 + el2 + n2  """
def loss_photonqudit_el1el2n2(rho, eff):
        """ Add linear loss for a time-bin photonic qudit tensored with a three qubits el1 + + el2 + n2 (Si29) """

        a_1_5 = qt.tensor(qt.destroy(N), IdN, IdN, IdN, IdN)
        a_5_5 = qt.tensor(IdN, IdN, IdN, IdN, qt.destroy(N))

        bs_loss_per_timebin = general_BS(1j*np.sqrt(eff), np.sqrt(1 -eff), a_1_5, a_5_5)
        oper_loss = qt.tensor(Id2, Id2, Id2, bs_loss_per_timebin)
        # print('number of photons per timebin before loss  =', (Noperator*rho.ptrace([3])).tr(), (Noperator*rho.ptrace([4])).tr(), (Noperator*rho.ptrace([5])).tr(), (Noperator*rho.ptrace([6])).tr())

        for i in range(4):
            #operation of BS1 50/50 on early reflected beam
            rho_1 = (oper_loss*(qt.tensor(rho, qt.fock_dm(N, 0)))*oper_loss.dag()).ptrace([0, 1, 2, 4, 5, 6, 7])
            rho = rho_1
            # print('number of photons per timebin after loss  =', (Noperator*rho.ptrace([3])).tr(), (Noperator*rho.ptrace([4])).tr(), (Noperator*rho.ptrace([5])).tr(), (Noperator*rho.ptrace([6])).tr())


        return rho

""" To figure out correlations between each time bin and matter qubit spins """
def qudit_collapse(rho, timebine):
    # for testing only: measure the correlatikos of spin state with each time bin
    Pj_1000 = qt.composite(Id2, Id2, qt.ket2dm(qt.fock(N, 1)), qt.ket2dm(qt.fock(N, 0)), qt.ket2dm(qt.fock(N, 0)), qt.ket2dm(qt.fock(N, 0)))
    Pj_0100 = qt.composite(Id2, Id2, qt.ket2dm(qt.fock(N, 0)), qt.ket2dm(qt.fock(N, 1)), qt.ket2dm(qt.fock(N, 0)), qt.ket2dm(qt.fock(N, 0)))        
    Pj_0010 = qt.composite(Id2, Id2, qt.ket2dm(qt.fock(N, 0)), qt.ket2dm(qt.fock(N, 0)), qt.ket2dm(qt.fock(N, 1)), qt.ket2dm(qt.fock(N, 0)))
    Pj_0001 = qt.composite(Id2, Id2,qt.ket2dm(qt.fock(N, 0)), qt.ket2dm(qt.fock(N, 0)), qt.ket2dm(qt.fock(N, 0)), qt.ket2dm(qt.fock(N, 1)))        
    # probability of each bin firing
    brate_bin_1 = (Pj_1000*rho*Pj_1000.dag()).tr()
    brate_bin_2 = (Pj_0100*rho*Pj_0100.dag()).tr()
    brate_bin_3 = (Pj_0010*rho*Pj_0010.dag()).tr()
    brate_bin_4 = (Pj_0001*rho*Pj_0001.dag()).tr()    
  
    #Final density matrix of the electron state
    rho_final_bin_1 = ((Pj_1000*rho*Pj_1000.dag())/brate_bin_1).ptrace([0, 1]) # spin state left over after bin 1 firing
    rho_final_bin_2 = ((Pj_0100*rho*Pj_0100.dag())/brate_bin_2).ptrace([0, 1]) # spin state left over after bin 2 firing
    rho_final_bin_3 = ((Pj_0010*rho*Pj_0010.dag())/brate_bin_3).ptrace([0, 1]) # spin state left over after bin 3 firing
    rho_final_bin_4 = ((Pj_0001*rho*Pj_0001.dag())/brate_bin_4).ptrace([0, 1]) # spin state left over after bin 4 firing
    
    if timebine == 0:
        rho = rho_final_bin_1
    elif timebine == 1:
        rho = rho_final_bin_2
    elif timebine == 2:
        rho = rho_final_bin_3
    elif timebine == 3:
        rho = rho_final_bin_4
    return rho

""" Generate d = 4 qudit with the timebin encoding """

def generate_qudit(alpha):
        "generate qudit with equal amplitude time bins and normalized with mu number of photons per qudit"
        a = b = c = d = 1
        norm = np.sqrt(4)
        
        time_bin_0 = qt.tensor(qt.coherent(N, alpha*norm*a), qt.coherent(N, 0), qt.coherent(N, 0), qt.coherent(N, 0))
        time_bin_1 = qt.tensor(qt.coherent(N, 0), qt.coherent(N, alpha*norm*b), qt.coherent(N, 0), qt.coherent(N, 0))
        time_bin_2 = qt.tensor(qt.coherent(N, 0), qt.coherent(N, 0), qt.coherent(N, alpha*norm*c), qt.coherent(N, 0))
        time_bin_3 = qt.tensor(qt.coherent(N, 0), qt.coherent(N, 0), qt.coherent(N, 0), qt.coherent(N, alpha*norm*d))
        qudit = (time_bin_0 + time_bin_1 + time_bin_2 + time_bin_3).unit()
        
        return qudit
    