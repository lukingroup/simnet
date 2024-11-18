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
    
    def parallel_entanglement_1timebin_ee(self, e1_initial, e2_initial, imperfections_1, imperfections_2, mu1, mu2, bin = 1):
        """ El-el parallel entanglement """

        if imperfections_1['contrast_noise'] == False:
            wl_1 = self.fiber_network.siv1.optimum_freq
            cav_refl_1 = self.fiber_network.siv1.cav_refl(wl_1)
                
        elif imperfections_1['contrast_noise'] == True:
            wl_1 = np.random.normal(loc=self.fiber_network.siv1.optimum_freq, scale=50)
            cav_refl_1 = self.fiber_network.siv1.cav_refl(wl_1)

        if imperfections_2['contrast_noise'] == False:
            wl_2 = self.fiber_network.siv2.optimum_freq
            cav_refl_2 = self.fiber_network.siv2.cav_refl(wl_2)
                
        elif imperfections_2['contrast_noise'] == True:
            wl_2 = np.random.normal(loc=self.fiber_network.siv2.optimum_freq, scale=50)
            cav_refl_2 = self.fiber_network.siv2.cav_refl(wl_2)


        fidel_values_pi_pi2_1 = {'pi': imperfections_1['mw_fid_num'][0],
                    'pi_half': imperfections_1['mw_fid_num'][1]
                    }        
        fidel_values_pi_pi2_2 = {'pi': imperfections_2['mw_fid_num'][0],
                    'pi_half': imperfections_2['mw_fid_num'][1]
                    }        

        # 'real'/'perfect' and 'stable'/'noisy'
        gates1 = set_mw_fidelities(fid = imperfections_1['mw'], noise = imperfections_1['mw_noise'], fidel_val = fidel_values_pi_pi2_1)
        gates2 = set_mw_fidelities(fid = imperfections_2['mw'], noise = imperfections_2['mw_noise'], fidel_val = fidel_values_pi_pi2_2)


        siv_beamsplitters1 = siv_beamsplitter_1timebin(cav_refl_1, imperfections_1['contrast'])
        siv_beamsplitters2 = siv_beamsplitter_1timebin(cav_refl_2, imperfections_2['contrast'])


        ## photonic qubit node 1
        alpha_1 = np.sqrt(mu1)
        input_coh_1 = qt.coherent(N, alpha_1)

        ## photonic qubit node 2
        alpha_2 = np.sqrt(mu2)
        input_coh_2 = qt.coherent(N, alpha_2)

        rho_0_1 = qt.tensor(e1_initial, qt.ket2dm(input_coh_1))
        rho_0_2 = qt.tensor(e2_initial, qt.ket2dm(input_coh_2))

        # print('Initial number of photons per qubit =', (Noperator*rho_0.ptrace([1])).tr(),  (Noperator*rho_0.ptrace([2])).tr())

        ## First Node

        # reflect
        rho_1_1 = siv_beamsplitters1[0]*(qt.tensor(rho_0_1, qt.fock_dm(N, 0)))*siv_beamsplitters1[0].dag()
        rho_2_1 = (siv_beamsplitters1[1]*(qt.tensor(rho_1_1, qt.fock_dm(N, 0)))*siv_beamsplitters1[1].dag()).ptrace([0, 2])

        # # print('The number of photons mid spin photon =', (Noperator*rho_2.ptrace([1])).tr(), (Noperator*rho_2.ptrace([2])).tr())

        ## Link loss (add link loss )

        eff_1 = g12_b16_network.link_eff_1
        rho_3_1 = loss_photonqubit_1timebin(rho_2_1, eff_1)

        ## Second Node

        # reflect
        rho_1_2 = siv_beamsplitters2[0]*(qt.tensor(rho_0_2, qt.fock_dm(N, 0)))*siv_beamsplitters2[0].dag()
        rho_2_2 = (siv_beamsplitters2[1]*(qt.tensor(rho_1_2, qt.fock_dm(N, 0)))*siv_beamsplitters2[1].dag()).ptrace([0, 2])

        ## Link loss (add link loss )

        eff_2 = g12_b16_network.link_eff_2
        rho_3_2 = loss_photonqubit_1timebin(rho_2_2, eff_2)

        ## Combine the states
        rho_tot = qt.tensor(rho_3_1, rho_3_2)

        ## Measure photon in X basis
        phi = 0
        # pick whether you want to always get a specific measurement result prob = "deterministic" or
        # prob = "probabilistic" which is probilistic output like in the experiment
        rho_15 = phi_photon_measurement_1timebin_ee_parallel(rho_tot, phi, bin, tdi_noise = imperfections_1['tdinoise'], prob = "deterministic")

        return rho_15
    
    def parallel_entanglement_2timebins_ee(self, e1_initial, e2_initial, imperfections_1, imperfections_2, mu1, mu2, bin = 1):
        """ El-el parallel entanglement """

        if imperfections_1['contrast_noise'] == False:
            wl_1 = self.fiber_network.siv1.optimum_freq
            cav_refl_1 = self.fiber_network.siv1.cav_refl(wl_1)
                
        elif imperfections_1['contrast_noise'] == True:
            wl_1 = np.random.normal(loc=self.fiber_network.siv1.optimum_freq, scale=50)
            cav_refl_1 = self.fiber_network.siv1.cav_refl(wl_1)

        if imperfections_2['contrast_noise'] == False:
            wl_2 = self.fiber_network.siv2.optimum_freq
            cav_refl_2 = self.fiber_network.siv2.cav_refl(wl_2)
                
        elif imperfections_2['contrast_noise'] == True:
            wl_2 = np.random.normal(loc=self.fiber_network.siv2.optimum_freq, scale=50)
            cav_refl_2 = self.fiber_network.siv2.cav_refl(wl_2)

        
        fidel_values_pi_pi2_1 = {'pi': imperfections_1['mw_fid_num'][0],
                    'pi_half': imperfections_1['mw_fid_num'][1]
                    }        
        fidel_values_pi_pi2_2 = {'pi': imperfections_2['mw_fid_num'][0],
                    'pi_half': imperfections_2['mw_fid_num'][1]
                    }        

        # 'real'/'perfect' and 'stable'/'noisy'
        gates1 = set_mw_fidelities(fid = imperfections_1['mw'], noise = imperfections_1['mw_noise'], fidel_val = fidel_values_pi_pi2_1)
        gates2 = set_mw_fidelities(fid = imperfections_2['mw'], noise = imperfections_2['mw_noise'], fidel_val = fidel_values_pi_pi2_2)

        
        siv_beamsplitters1 = siv_beamsplitter(cav_refl_1, imperfections_1['contrast'])
        siv_beamsplitters2 = siv_beamsplitter(cav_refl_2, imperfections_2['contrast'])


        ## photonic qubit node 1
        alpha_1 = np.sqrt(mu1)
        early_time_bin_1 = qt.tensor(qt.coherent(N, alpha_1/np.sqrt(2)), qt.coherent(N, 0))
        late_time_bin_1 = qt.tensor(qt.coherent(N, 0), qt.coherent(N, alpha_1/np.sqrt(2)))
        input_coh_1 = (early_time_bin_1 + late_time_bin_1)
        
        ## photonic qubit node 2
        alpha_2 = np.sqrt(mu2)
        early_time_bin_2 = qt.tensor(qt.coherent(N, alpha_2/np.sqrt(2)), qt.coherent(N, 0))
        late_time_bin_2 = qt.tensor(qt.coherent(N, 0), qt.coherent(N, alpha_2/np.sqrt(2)))
        input_coh_2 = (early_time_bin_2 + late_time_bin_2)

        rho_0_1 = qt.tensor(e1_initial, qt.ket2dm(input_coh_1))
        rho_0_2 = qt.tensor(e2_initial, qt.ket2dm(input_coh_2))

        # print('Initial number of photons per qubit =', (Noperator*rho_0.ptrace([1])).tr(),  (Noperator*rho_0.ptrace([2])).tr())

        ## First Node
        
        # reflect early
        rho_1_1 = siv_beamsplitters1[0]*(qt.tensor(rho_0_1, qt.fock_dm(N, 0)))*siv_beamsplitters1[0].dag()
        rho_2_1 = (siv_beamsplitters1[1]*(qt.tensor(rho_1_1, qt.fock_dm(N, 0)))*siv_beamsplitters1[1].dag()).ptrace([0, 2, 3])

        # do a pi gate on the electron
        pi_oper_1 = qt.tensor(gates1['pi'], IdN, IdN)
        rho_3_1 = pi_oper_1*rho_2_1*pi_oper_1.dag()
        
        # # print('The number of photons mid spin photon =', (Noperator*rho_2.ptrace([1])).tr(), (Noperator*rho_2.ptrace([2])).tr())

        # reflect late
        rho_4_1 = siv_beamsplitters1[0]*(qt.tensor(rho_3_1, qt.fock_dm(N, 0)))*siv_beamsplitters1[0].dag()
        rho_5_1 = (siv_beamsplitters1[1]*(qt.tensor(rho_4_1, qt.fock_dm(N, 0)))*siv_beamsplitters1[1].dag()).ptrace([0, 2, 3])
        
        rho_6_1 = pi_oper_1*rho_5_1*pi_oper_1.dag()

        ## Link loss (add link loss )

        eff_1 = g12_b16_network.link_eff_1
        rho_7_1 = loss_photonqubit_elSpin(rho_6_1, eff_1)

        ## Second Node
        
        # reflect early
        rho_1_2 = siv_beamsplitters2[0]*(qt.tensor(rho_0_2, qt.fock_dm(N, 0)))*siv_beamsplitters2[0].dag()
        rho_2_2 = (siv_beamsplitters2[1]*(qt.tensor(rho_1_2, qt.fock_dm(N, 0)))*siv_beamsplitters2[1].dag()).ptrace([0, 2, 3])

        # do a pi gate on the electron
        pi_oper_2 = qt.tensor(gates2['pi'], IdN, IdN)
        rho_3_2 = pi_oper_2*rho_2_2*pi_oper_2.dag()
        
        # # print('The number of photons mid spin photon =', (Noperator*rho_2.ptrace([1])).tr(), (Noperator*rho_2.ptrace([2])).tr())

        # reflect late
        rho_4_2 = siv_beamsplitters2[0]*(qt.tensor(rho_3_2, qt.fock_dm(N, 0)))*siv_beamsplitters2[0].dag()
        rho_5_2 = (siv_beamsplitters2[1]*(qt.tensor(rho_4_2, qt.fock_dm(N, 0)))*siv_beamsplitters2[1].dag()).ptrace([0, 2, 3])
        
        rho_6_2 = pi_oper_2*rho_5_2*pi_oper_2.dag()

        ## Link loss (add link loss )

        eff_2 = g12_b16_network.link_eff_2
        rho_7_2 = loss_photonqubit_elSpin(rho_6_2, eff_2)
        
        ## Combine the states
        rho_tot = qt.tensor(rho_7_1, rho_7_2)

        ## Measure photon in X basis
        phi = 0
        # pick whether you want to always get a specific measurement result prob = "deterministic" or
        # prob = "probabilistic" which is probilistic output like in the experiment
        rho_15 = phi_photon_measurement_ee_parallel(rho_tot, phi, bin, tdi_noise = imperfections_1['tdinoise'], prob = "deterministic")

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
        
def loss_photonqubit_1timebin(rho, eff):

    bs_e = general_BS(1j*np.sqrt(1 - eff), np.sqrt(eff), a_1_2, a_2_2)

     #operation of BS1 50/50 on early reflected beam
    oper_e = qt.tensor(Id2, bs_e)
    
    rho_1 = (oper_e*(qt.tensor(rho, qt.fock_dm(N, 0)))*oper_e.dag()).ptrace([0, 1])

    # print('The number of photons after loss =', (Noperator*rho_2.ptrace([1])).tr() + (Noperator*rho_2.ptrace([2])).tr())

    return rho_1


def siv_beamsplitter_1timebin(cav_refl, contrast):

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
    bs1_up = (((-1j*phase1_up_1/2)*(a_1_2.dag()*a_1_2 - a_2_2.dag()*a_2_2)).expm()*\
                (-theta1_up*(a_1_2.dag()*a_2_2 - a_2_2.dag()*a_1_2)).expm()*\
                ((-1j*phase1_up_2/2)*(a_1_2.dag()*a_1_2 - a_2_2.dag()*a_2_2)).expm())

    bs1_down = (((-1j*phase1_down_1/2)*(a_1_2.dag()*a_1_2 - a_2_2.dag()*a_2_2)).expm()*\
                (-theta1_down*(a_1_2.dag()*a_2_2 - a_2_2.dag()*a_1_2)).expm()*\
                ((-1j*phase1_down_2/2)*(a_1_2.dag()*a_1_2 - a_2_2.dag()*a_2_2)).expm())

    bs2_up = general_BS(sc, transm, a_1_3, a_3_3)
    bs2_down = general_BS(nsc, ntransm, a_1_3, a_3_3)

    #operation to reflect early bin
    oper1 = qt.tensor(qt.ket2dm(qt.basis(2, 0)), bs1_up) + qt.tensor(qt.ket2dm(qt.basis(2, 1)), bs1_down)
    #operation to split transmission and scattering for early bin
    oper2 = qt.tensor(qt.ket2dm(qt.basis(2, 0)),bs2_up) + qt.tensor(qt.ket2dm(qt.basis(2, 1)),bs2_down)
    return oper1, oper2

def phi_photon_measurement_1timebin_ee_parallel(rho, phi, bin, tdi_noise = imperfections_1['tdinoise'], prob = "prob"):

    # the combine state of el-el and two qubits is el-e-l-el-e-l so we need to redo the raising/lowering operators
    
    # loss
    a_1 = qt.tensor(qt.destroy(N), Id2, IdN)
    a_2 = qt.tensor(IdN, Id2, qt.destroy(N))
   

    Pj_10 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 1)),  Id2, qt.ket2dm(qt.basis(N, 0)))
    Pj_01 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 0)),  Id2, qt.ket2dm(qt.basis(N, 1)))        
    

    ratio = np.random.normal(loc=0.5, scale=0*0.1*0.5)
    # pick which TDI to use 
    # short tdi entangles
    angle = np.random.normal(loc=2*np.pi + tdi_noise, scale=0*0.1*2*np.pi)
    r = np.exp(1j*(angle))*np.sqrt(ratio)
    if np.abs(r) > 1:
        r = 1
    bs_5050_el_r = general_BS(r, np.sqrt(1-(abs(r))**2), a_1, a_2) 
    
    oper = qt.tensor(Id2, bs_5050_el_r)
    ## now the qudit is rotated into the basis we want to measure
    rho1 = oper*rho*oper.dag()
    
    # probability of each bin firing
    brate_bin_1 = (Pj_10*rho1*Pj_10.dag()).tr()
    brate_bin_2 = (Pj_01*rho1*Pj_01.dag()).tr()
    
    bnorm_tot_rates = brate_bin_1 + brate_bin_2
    
    # probabilities
    brate_bin_1_norm = brate_bin_1/bnorm_tot_rates
    brate_bin_2_norm = brate_bin_2/bnorm_tot_rates
   
    #Final density matrix of the electron state
    rho_final_bin_1 = ((Pj_10*rho1*Pj_10.dag())/brate_bin_1).ptrace([0, 2]) # spin state left over after bin 1 firing
    rho_final_bin_2 = ((Pj_01*rho1*Pj_01.dag())/brate_bin_2).ptrace([0, 2]) # spin state left over after bin 2 firing

    
    if prob == "deterministic":
        if bin == 1:
            spin_state = rho_final_bin_1
        elif bin == 2:
            spin_state = rho_final_bin_2
        output = [spin_state, brate_bin_1, brate_bin_2]

    elif prob == "probabilistic":
        # probabilistic projective measurement
        quantum_measurement_s1s2 = np.random.choice([1, 2], p=[brate_bin_1_norm, brate_bin_2_norm])
        if quantum_measurement_s1s2 == 1:
            spin_state = rho_final_bin_1
        elif quantum_measurement_s1s2 == 2:
            spin_state = rho_final_bin_2
        output = [spin_state, brate_bin_1, brate_bin_2, quantum_measurement_s1s2]
    return output

def phi_photon_measurement_ee_parallel(rho, phi, bin, tdi_noise = imperfections_1['tdinoise'], prob = "prob"):

    # the combine state of el-el and two qubits is el-e-l-el-e-l so we need to redo the raising/lowering operators
    
    # loss
    a_1_4 = qt.tensor(qt.destroy(N), IdN, Id2, IdN, IdN)
    a_2_4 = qt.tensor(IdN, qt.destroy(N), Id2, IdN, IdN)
    a_3_4 = qt.tensor(IdN, IdN, Id2, qt.destroy(N), IdN)
    a_4_4 = qt.tensor(IdN, IdN, Id2, IdN, qt.destroy(N))

    Pj_1000 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 1)), qt.ket2dm(qt.basis(N, 0)), Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 0)))
    Pj_0100 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 1)), Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 0)))        
    Pj_0010 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 0)), Id2, qt.ket2dm(qt.basis(N, 1)), qt.ket2dm(qt.basis(N, 0)))
    Pj_0001 = qt.composite(Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 0)), Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 1)))        


    ratio = np.random.normal(loc=0.5, scale=0*0.1*0.5)
    # pick which TDI to use 
    # short tdi entangles
    angle = np.random.normal(loc=2*np.pi + tdi_noise, scale=0*0.1*2*np.pi)
    r = np.exp(1j*(angle))*np.sqrt(ratio)
    if np.abs(r) > 1:
        r = 1
    bs_5050_el_r_01 = general_BS(r, np.sqrt(1-(abs(r))**2), a_1_4, a_3_4) # earlies
    bs_5050_el_r_02 = general_BS(r, np.sqrt(1-(abs(r))**2), a_2_4, a_4_4) # lates
    oper_1 = qt.tensor(Id2, bs_5050_el_r_01)
    oper_2 = qt.tensor(Id2, bs_5050_el_r_02)
    ## now the qudit is rotated into the basis we want to measure
    rho1 = oper_1*rho*oper_1.dag()
    rho2 = oper_2*rho1*oper_2.dag()
    
    # probability of each bin firing
    brate_bin_1 = (Pj_1000*rho2*Pj_1000.dag()).tr()
    brate_bin_2 = (Pj_0100*rho2*Pj_0100.dag()).tr()
    brate_bin_3 = (Pj_0010*rho2*Pj_0010.dag()).tr()
    brate_bin_4 = (Pj_0001*rho2*Pj_0001.dag()).tr()
    bnorm_tot_rates = brate_bin_1 + brate_bin_2 + brate_bin_3 + brate_bin_4
    
    # probabilities
    brate_bin_1_norm = brate_bin_1/bnorm_tot_rates
    brate_bin_2_norm = brate_bin_2/bnorm_tot_rates
    brate_bin_3_norm = brate_bin_3/bnorm_tot_rates
    brate_bin_4_norm = brate_bin_4/bnorm_tot_rates

    #Final density matrix of the electron state
    rho_final_bin_1 = ((Pj_1000*rho2*Pj_1000.dag())/brate_bin_1).ptrace([0, 3]) # spin state left over after bin 1 firing
    rho_final_bin_2 = ((Pj_0100*rho2*Pj_0100.dag())/brate_bin_2).ptrace([0, 3]) # spin state left over after bin 2 firing
    rho_final_bin_3 = ((Pj_0010*rho2*Pj_0010.dag())/brate_bin_3).ptrace([0, 3]) # spin state left over after bin 3 firing
    rho_final_bin_4 = ((Pj_0001*rho2*Pj_0001.dag())/brate_bin_4).ptrace([0, 3]) # spin state left over after bin 4 firing

    
    if prob == "deterministic":
        if bin == 1:
            spin_state = rho_final_bin_1
        elif bin == 2:
            spin_state = rho_final_bin_2
        elif bin == 3:
            spin_state = rho_final_bin_3
        elif bin == 4:
            spin_state = rho_final_bin_4 
        output = [spin_state, brate_bin_1, brate_bin_2, brate_bin_3, brate_bin_4]

    elif prob == "probabilistic":
        # probabilistic projective measurement
        quantum_measurement_s1s2 = np.random.choice([1, 2, 3, 4], p=[brate_bin_1_norm, brate_bin_2_norm, brate_bin_3_norm, brate_bin_4_norm])
        if quantum_measurement_s1s2 == 1:
            spin_state = rho_final_bin_1
        elif quantum_measurement_s1s2 == 2:
            spin_state = rho_final_bin_2
        elif quantum_measurement_s1s2 == 3:
            spin_state = rho_final_bin_3
        elif quantum_measurement_s1s2 == 4:
            spin_state = rho_final_bin_4 
        output = [spin_state, brate_bin_1, brate_bin_2, brate_bin_3, brate_bin_4, quantum_measurement_s1s2]
    return output

