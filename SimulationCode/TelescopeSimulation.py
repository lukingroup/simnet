#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:30:05 2023

@author: azizasuleymanzade
"""

from fiber_network import FiberNetwork
from SiVnodes import SiV
import numpy as np
from SiVgates import *
from Plots import *
import qutip as qt
  
class TelescopeSimulation:
    def __init__(self, fiber_network):
        """Initialize the simulation with an initial state."""
        self.fiber_network = fiber_network

    def serial_entanglement_ee(self, ee_initial, imperfections, mu):
        """ El-el serial entanglmenet """

        if imperfections['contrast_noise'] == False:
            wl_1 = self.fiber_network.siv1.optimum_freq
            wl_2 = self.fiber_network.siv2.optimum_freq
            cav_refl_1 = self.fiber_network.siv1.cav_refl(wl_1)
            cav_refl_2 = self.fiber_network.siv2.cav_refl(wl_2)
                
        elif imperfections['contrast_noise'] == True:
            wl_1 = np.random.normal(loc=self.fiber_network.siv1.optimum_freq, scale=50)
            wl_2 = np.random.normal(loc=self.fiber_network.siv2.optimum_freq, scale=50)
            cav_refl_1 = self.fiber_network.siv1.cav_refl(wl_1)
            cav_refl_2 = self.fiber_network.siv2.cav_refl(wl_2)

        # I will assume the microwave fidelities are the same in both nodes
        fidel_values_pi_pi2 = {'pi': imperfections['mw_fid_num'][0],
                    'pi_half': imperfections['mw_fid_num'][1]
                    }        

        # 'real'/'perfect' and 'stable'/'noisy'
        gates = set_mw_fidelities(fid = imperfections['mw'], noise = imperfections['mw_noise'], fidel_val = fidel_values_pi_pi2)
        siv_beamsplitters1 = siv_beamsplitter_ee_e1_serial(cav_refl_1, imperfections['contrast'])
        siv_beamsplitters2 = siv_beamsplitter_ee_e2_serial(cav_refl_2, imperfections['contrast'])

        alpha = np.sqrt(mu)
        early_time_bin = qt.tensor(qt.coherent(N, alpha/np.sqrt(2)), qt.coherent(N, 0))
        late_time_bin = qt.tensor(qt.coherent(N, 0), qt.coherent(N, alpha/np.sqrt(2)))
        input_coh = (early_time_bin + late_time_bin)
        rho_0 = qt.tensor(ee_initial, qt.ket2dm(input_coh))

        # print('Initial number of photons per qubit =', (Noperator*rho_0.ptrace([1])).tr(),  (Noperator*rho_0.ptrace([2])).tr())

        ## First Node
        
        # reflect early
        rho_1 = siv_beamsplitters1[0]*(qt.tensor(rho_0, qt.fock_dm(N, 0)))*siv_beamsplitters1[0].dag()
        rho_2 = (siv_beamsplitters1[1]*(qt.tensor(rho_1, qt.fock_dm(N, 0)))*siv_beamsplitters1[1].dag()).ptrace([0, 1, 3, 4])

        # do a pi gate on the electron
        pi_oper = qt.tensor(gates['pi'], Id2, IdN, IdN)
        rho_3 = pi_oper*rho_2*pi_oper.dag()
        
        # print('The number of photons mid spin photon =', (Noperator*rho_2.ptrace([1])).tr(), (Noperator*rho_2.ptrace([2])).tr())

        # reflect late
        rho_4 = siv_beamsplitters1[0]*(qt.tensor(rho_3, qt.fock_dm(N, 0)))*siv_beamsplitters1[0].dag()
        rho_5 = (siv_beamsplitters1[1]*(qt.tensor(rho_4, qt.fock_dm(N, 0)))*siv_beamsplitters1[1].dag()).ptrace([0, 1, 3, 4])
        
        rho_6 = pi_oper*rho_5*pi_oper.dag()

        ## Link loss

        eff = self.fiber_network.link_efficiency
        rho_7 = loss_photonqubit_ee_serial(rho_6, eff)

        ## Second Node

        # reflect early
        rho_8 = siv_beamsplitters2[0]*(qt.tensor(rho_7, qt.fock_dm(N, 0)))*siv_beamsplitters2[0].dag()
        rho_9 = (siv_beamsplitters2[1]*(qt.tensor(rho_8, qt.fock_dm(N, 0)))*siv_beamsplitters2[1].dag()).ptrace([0, 1, 3, 4])

        # do a pi gate on the electron
        pi_oper = qt.tensor(Id2, gates['pi'], IdN, IdN)
        rho_10 = pi_oper*rho_9*pi_oper.dag()
        
        # print('The number of photons mid spin photon =', (Noperator*rho_2.ptrace([1])).tr(), (Noperator*rho_2.ptrace([2])).tr())

        # reflect late
        rho_11 = siv_beamsplitters2[0]*(qt.tensor(rho_10, qt.fock_dm(N, 0)))*siv_beamsplitters2[0].dag()
        rho_12 = (siv_beamsplitters2[1]*(qt.tensor(rho_11, qt.fock_dm(N, 0)))*siv_beamsplitters2[1].dag()).ptrace([0, 1, 3, 4])
        
        rho_13 = pi_oper*rho_12*pi_oper.dag()

        ## Detection loss

        eff = self.fiber_network.detection_eff
        rho_14 = loss_photonqubit_ee_serial(rho_13, eff)

        ## Measure photon in X basis
        phi = 0
        rho_15 = phi_photon_measurement_ee_serial(rho_14, phi, tdi_noise = imperfections['tdinoise'])

        return rho_15
    
    def single_node_electron_photon_entanglement(self, el_initial, imperfections, mu):
        ## add snspd loss (somehow after interfering)

        if imperfections['contrast_noise'] == True:
            wl = self.fiber_network.siv1.optimum_freq
            cav_refl = self.fiber_network.siv1.cav_refl(wl)
        elif imperfections['contrast_noise'] == False:
            wl = np.random.normal(loc=self.fiber_network.siv1.optimum_freq, scale=50)
            cav_refl = self.fiber_network.siv1.cav_refl(wl)

        siv_beamsplitters = siv_beamsplitter(cav_refl, imperfections['contrast'])
        gates = mw_gates('perfect', 'perfect')

        """Spin photon entnaglement"""
        rho_1 = spin_photon_entaglement(siv_beamsplitters, el_initial, gates['pi'], mu)

        """Add eff if needed"""
        eff = 1
        rho_1l = loss_photonqubit_elSpin(rho_1, eff)
        
        """Tomography"""
        outputZZ = el_photon_bell_state_Ztomography(rho_1l)
        outputXX = el_photon_bell_state_Xtomography(rho_1l)
        outputYY = el_photon_bell_state_Ytomography(rho_1l)
        
        """Plot"""
        bell_state_barplotZZ(outputZZ)
        bell_state_barplotXX(outputXX)
        bell_state_barplotYY(outputYY)
        
        return 

    def single_node_electron_efficiency(self, el_initial, imperfections, mu):
        """Start the simulation."""
        print("Simulation at mu = :", mu)

        if imperfections['contrast_noise'] == True:
            wl = self.fiber_network.siv1.optimum_freq
            cav_refl = self.fiber_network.siv1.cav_refl(wl)
        elif imperfections['contrast_noise'] == False:
            wl = np.random.normal(loc=self.fiber_network.siv1.optimum_freq, scale=50)
            cav_refl = self.fiber_network.siv1.cav_refl(wl)

        siv_beamsplitters = siv_beamsplitter(cav_refl, imperfections['contrast'])
        gates = mw_gates('perfect', 'perfect')

        """Spin photon entnaglement"""
        rho_1 = spin_photon_entaglement(siv_beamsplitters, el_initial, gates['pi'], mu)
       
        """pi half on the electron"""
        oper_pi_half = qt.tensor(gates['pi_half'], IdN, IdN)
        rho_2 = oper_pi_half*rho_1*oper_pi_half.dag()

        """Add eff if needed"""
        eff = 1
        rho_2l = loss_photonqubit_elSpin(rho_2, eff)
        
        """Measure number of photons"""
        Nearly = (Noperator*rho_2l.ptrace([1])).tr()
        Nlate =  (Noperator*rho_2l.ptrace([2])).tr()
        Ntot = Nearly + Nlate 
                    
        rho_el_final = (rho_2l.ptrace([0])).unit()
        fid = qt.fidelity(rho_el_final, rho_ideal_Zp)**2    

        return Ntot, fid

    def single_node_el_si29_amplitude(self, el_initial, si29_initial, imperfections, mu, mu_LO, phi_LO):
        
        """Start the sensitivity measurement."""
        print("Simulation at mu = :", mu, ' mu_LO = ', mu_LO, ' and phi_LO = ', phi_LO )

        if imperfections['contrast_noise'] == False:
            wl = self.fiber_network.siv1.optimum_freq
            cav_refl = self.fiber_network.siv1.cav_refl(wl)
        elif imperfections['contrast_noise'] == True:
            wl = np.random.normal(loc=self.fiber_network.siv1.optimum_freq, scale=50)
            cav_refl = self.fiber_network.siv1.cav_refl(wl)

        siv_beamsplitters = siv_beamsplitter_si29(cav_refl, imperfections['contrast'])
        gates = mw_gates('perfect', 'perfect')
        cond_gates = cond_mw_gates(gates)

        """Nucleus photon entanglement"""
        rho_1 = nucleus_photon_entaglement(siv_beamsplitters, el_initial, si29_initial, cond_gates['pi_mw1'], mu)
        
        # print('rho_1', rho_1.ptrace([1]))

        """pi half on the electron"""
        oper_pi_half = qt.tensor(gates['pi_half'], Id2, IdN, IdN)
        rho_2 = oper_pi_half*rho_1*oper_pi_half.dag()
        
        # print('rho_2', rho_2.ptrace([1]))

        """Interfere photons with LO first and count (then do postselection on heralding through electron)"""
        counts, rho_el_si29 = interfere_qubit_with_LO(rho_2, mu_LO, phi_LO)
        
        # print('rho_el_si29', rho_el_si29.ptrace([1]))

        """Photon counting strategy output: an operator to apply to si29""" 
        oper_erasure_Si29 = erasure_strategy2(counts)

        """Heralding the arrival of the photon using the electron state, down is the correct one (2, 0)"""
        Pe_down = qt.composite(qt.ket2dm(qt.basis(2,1)), Id2)
        eff_herald = (Pe_down*rho_el_si29*Pe_down.dag()).tr()

        # print((Pe_down*rho_el_si29*Pe_down.dag()).ptrace([1]))

        # print('heralding efficiency', eff_herald)
        rho_heralded = ((Pe_down*rho_el_si29*Pe_down.dag())/eff_herald).ptrace([1])
        print(rho_heralded)
        """Apply the strategy operator on si29"""
        rho_si29_final = oper_erasure_Si29*rho_heralded*oper_erasure_Si29.dag()
        print(rho_si29_final)

        """Measure the sigma_z of si29"""
        population_z_0 = qt.expect(qt.sigmax(), rho_si29_final)
        
        print(qt.expect(qt.sigmax(), rho_si29_final))

        return population_z_0

    def plot_SiV(self):
        """Plotting Function"""

""" Supporting functions """
def erasure_strategy2(counts):

    if counts['counts_early_1'] > counts['counts_early_2']:
        if counts['counts_late_1'] > counts['counts_late_2']:
            return Id2
        elif counts['counts_late_1'] < counts['counts_late_2']:
            return qt.sigmax
    elif counts['counts_early_1'] < counts['counts_early_2']:
        if counts['counts_late_1'] > counts['counts_late_2']:
            return qt.sigmax
        elif counts['counts_late_1'] < counts['counts_late_2']:
            return Id2

# # Create SiVs:
# siv_a = SiV(kappa_in= 101*(10**3), kappa_w= (29)*(10**3), g=3.19*(10**3), wCav = (0)*(10**3), wSiv = 55*(10**3), dwEl = 0.5*(10**3)) # B16
# # Create Networks:
# b16_network = FiberNetwork(siv_a)

# # Create Simulation:
# sim = TelescopeSimulation(b16_network)

# imperfections ={'contrast_noise': False,
#                 'contrast': 'perfect'
#                }

# el_initial  = qt.ket2dm((qt.basis(2,0)+ qt.basis(2,1)).unit())
# si29_initial  = qt.ket2dm((qt.basis(2,0)+ qt.basis(2,1)).unit())
# mu = 0.01
# mu_LO = 0.01 ## alpha^2 per timebin (not both)
# phi_LO = 0
# #sim.single_node_electron_efficiency(el_initial, imperfections, mu)

# sigmaz_pop = sim.single_node_el_si29_amplitude(el_initial, si29_initial, imperfections, mu, mu_LO, phi_LO)
# print(abs(sigmaz_pop))
