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
import pandas as pd
import datetime
from qutip.qip.operations import *
from qutip.qip.circuit import *


import qutip as qt
  
class BlindComputing:
    def __init__(self, fiber_network):
        """Initialize the simulation with an initial state."""
        self.fiber_network = fiber_network
    
    def single_node_electron_photon_entanglement(self, el_initial, imperfections, mu):
        ## add snspd loss (somehow after interfering)
        wl = np.random.normal(loc=self.fiber_network.siv1.optimum_freq, scale=imperfections['contrast_noise']*50)
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

    def single_node_electron_exp(self, el_initial, imperfections, cluster_state_length, phi1, phi2, phi3, mu):
        """Start the simulation."""
        # print("Starting the simulation of the experimental sequence:")

        if imperfections['contrast_noise'] == False:
            wl = self.fiber_network.siv1.optimum_freq
            cav_refl = self.fiber_network.siv1.cav_refl(wl)
            
        elif imperfections['contrast_noise'] == True:
            wl = np.random.normal(loc=self.fiber_network.siv1.optimum_freq, scale=50)
            cav_refl = self.fiber_network.siv1.cav_refl(wl)

        siv_beamsplitters = siv_beamsplitter(cav_refl, imperfections['contrast'])

        fidel_values_pi_pi2 = {'pi': imperfections['mw_fid_num'][0],
                'pi_half': imperfections['mw_fid_num'][1]
                }
        
        # 'real'/'perfect' and 'stable'/'noisy'
        gates = set_mw_fidelities(fid = imperfections['mw'], noise = imperfections['mw_noise'], fidel_val = fidel_values_pi_pi2)
        """Initial state"""
        
        for i in np.arange(cluster_state_length):

            """Spin photon entanglement"""

            rho_1 = spin_photon_entaglement(siv_beamsplitters, el_initial, gates['pi'], mu)

            """ - pi on the electron, to associate up spin with early timebin"""
            oper_pi = qt.tensor(gates['pi'], IdN, IdN)
            rho_2 = oper_pi*rho_1*oper_pi.dag() 

            """Add losses if needed"""
            eff = self.fiber_network.detection_eff
            rho_2l = loss_photonqubit_elSpin(rho_2, eff)
            """Measure photon in the chosen basis"""
            rho_3 = phi_photon_measurement(rho_2l, phi1, imperfections['tdinoise'])

        return rho_3
    
    def single_rotation_electron(self, el_initial, imperfections, phi, mu):

        if imperfections['contrast_noise'] == False:
            wl = self.fiber_network.siv1.optimum_freq
            cav_refl = self.fiber_network.siv1.cav_refl(wl)
            
        elif imperfections['contrast_noise'] == True:
            wl = np.random.normal(loc=self.fiber_network.siv1.optimum_freq, scale=50)
            cav_refl = self.fiber_network.siv1.cav_refl(wl)

        siv_beamsplitters = siv_beamsplitter(cav_refl, imperfections['contrast'])

        fidel_values_pi_pi2 = {'pi': imperfections['mw_fid_num'][0],
                'pi_half': imperfections['mw_fid_num'][1]
                }
        
        # 'real'/'perfect' and 'stable'/'noisy'
        gates = set_mw_fidelities(fid = imperfections['mw'], noise = imperfections['mw_noise'], fidel_val = fidel_values_pi_pi2)

        """Spin photon entanglement"""

        rho_1 = spin_photon_entaglement(siv_beamsplitters, el_initial, gates['pi'], mu)

        """ - pi on the electron, to associate up spin with early timebin"""
        oper_pi = qt.tensor(gates['pi'], IdN, IdN)
        rho_2 = oper_pi*rho_1*oper_pi.dag() 

        """Add losses if needed"""
        eff = self.fiber_network.detection_eff
        rho_2l = loss_photonqubit_elSpin(rho_2, eff)
        
        """Measure photon in the chosen basis"""
        rho_3 = phi_photon_measurement(rho_2l, phi, imperfections['tdinoise'])

        return rho_3
    
    def single_qubit_universal_blind_gate_with_feedback(self, el_initial, imperfections, cluster_state_length, phi_array, mu):
        
        ## a gate that effectively creats a linear cluster state with the e on one end and  
        ## cluster_state_length number of photons, and measure these photons in phi_array angles
        ## this can perform arbitrary single qubit gates on the electron

        fidel_values_pi_pi2 = {'pi': imperfections['mw_fid_num'][0],
                    'pi_half': imperfections['mw_fid_num'][1]
                    }
        gates = set_mw_fidelities(fid = imperfections['mw'], noise = imperfections['mw_noise'], fidel_val = fidel_values_pi_pi2)
        
        # make a hadamart from an imperfect pi/2
        Had = 1j*rx(np.pi)*gates['pi_half']


        measurement_outputs = np.empty((0, 4), dtype=float)
        rho = el_initial
        mi = np.round(0, 1)
        for i in range(cluster_state_length): 
            ## active feedback on the angle
            angle =  ((-1)**(mi))*phi_array[i]
            rho_after_SPG = self.single_rotation_electron(rho, imperfections, angle, mu)
            # print(i, 'rho_after_SPG', rho_after_SPG)
            if i < 2:
                rho = Had*rho_after_SPG[0]*Had.dag()
                # print('after hadamard ', rho)
            else:
                rho = rho_after_SPG
                # print('raw final ', rho)

            mi = np.round(rho_after_SPG[1::],1)[0]
            measurement_outputs = np.array(np.vstack([measurement_outputs, [np.round(rho_after_SPG[1::],1)]]))
            
        
        rho_raw = rho[0]

        measurement_outputs = np.round(measurement_outputs[:,0], 1)
        rho_corrected = self.correction_single_qubit_universal_blind_gate_withfeedback(rho_raw, measurement_outputs)
        return rho_raw, rho_corrected, measurement_outputs

    def correction_single_qubit_universal_blind_gate_withfeedback(self, rho_raw, measurement_outputs):
    
        # The final correction for 3 rotations. Need to modify if the gate does more or less number of gates
        sz = measurement_outputs[0]+ measurement_outputs[2]
        sx = measurement_outputs[1]
        rho_corr = ((qt.sigmaz()**(sz))*(qt.sigmax()**sx)*rho_raw*((qt.sigmax()**sx).dag())*((qt.sigmaz()**(sz)).dag()))

        return rho_corr
    
    def phi_photon_measurement_withsi29(self, rho, phi, tdi_noise = 0):
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
    
    def two_qubit_intranode_blind_gate(self, eln_initial, imperfections, phi, mu):

        ## a gate that effectively creats entnagling logic or non entangling logic (product state)
        
        # first setup imperfect el rotations
        fidel_values_pi_pi2 = {'pi': imperfections['mw_fid_num'][0],
                    'pi_half': imperfections['mw_fid_num'][1]
                    }
        gates = set_mw_fidelities(fid = imperfections['mw'], noise = imperfections['mw_noise'], fidel_val = fidel_values_pi_pi2)
        
        pix = rz(-np.pi/2)*gates['pi']*rz(np.pi/2)
        # now add condition to our imperfect rotation (choose mw1 or mw2 depednign on what is the right condition)
        cond_pix = cond_mw_gates(pix)['pi_mw2']

        # apply a pi_x to electron if nucleus is in state 1
        rho1 =  cond_pix*eln_initial*cond_pix.dag()

        # apply a SPG on the electron
        rho2 = self.single_rotation_elsi29(rho1, imperfections, phi, mu)
        
        # apply a pi_x to electron if nucleus is in state 1
        rho =  cond_pix*rho2[0]*cond_pix.dag()

        # add a correction
        if rho2[1] == 1:
            correction_gate = qt.tensor(qt.sigmaz(),Id2)
        elif rho2[1] == 0:
            correction_gate = qt.tensor(Id2,qt.sigmaz())
        rho_corr = correction_gate*rho*correction_gate.dag()

        return rho, rho_corr, rho2[1::]
    
    def single_rotation_elsi29(self, rho, imperfections, phi, mu):

        if imperfections['contrast_noise'] == False:
            wl = self.fiber_network.siv1.optimum_freq
            cav_refl = self.fiber_network.siv1.cav_refl(wl)
            
        elif imperfections['contrast_noise'] == True:
            wl = np.random.normal(loc=self.fiber_network.siv1.optimum_freq, scale=50)
            cav_refl = self.fiber_network.siv1.cav_refl(wl)

        siv_beamsplitters = siv_beamsplitter_si29(cav_refl, imperfections['contrast'])

        fidel_values_pi_pi2 = {'pi': imperfections['mw_fid_num'][0],
                'pi_half': imperfections['mw_fid_num'][1]
                }        
        # 'real'/'perfect' and 'stable'/'noisy'
        gates = set_mw_fidelities(fid = imperfections['mw'], noise = imperfections['mw_noise'], fidel_val = fidel_values_pi_pi2)

        """Spin photon entanglement"""
        rho_1 = self.spin_photon_entaglement_withsi29(siv_beamsplitters, rho, gates['pi'], mu)

        """ - pi on the electron, to associate up spin with early timebin"""
        oper_pi = qt.tensor(gates['pi'], Id2, IdN, IdN)
        rho_2 = oper_pi*rho_1*oper_pi.dag() 

        """Add losses if needed"""
        eff = self.fiber_network.detection_eff
        rho_2l = loss_photonqubit_elSpin_withsi29(rho_2, eff)
        
        """Measure photon in the chosen basis"""
        rho_3 = phi_photon_measurement_withsi29(rho_2l, phi, imperfections['tdinoise'])

        return rho_3

    def el2_qudit_gate_QUBE(self, rho, gates, cav_refl, imperfections_contrast, mu):

        """ Electron 2 QUDIT Entaglement """

        # print('Initial number of photons per timebin and qubit =', (Noperator*rho.ptrace([3])).tr() + (Noperator*rho.ptrace([4])).tr() + (Noperator*rho.ptrace([5])).tr() + (Noperator*rho.ptrace([6])).tr())
        
        bs_QUBE = siv_beamsplitter_el2_qudit_QUBE(cav_refl, imperfections_contrast)

        pi_plus_y = gates['pi']
        pi_plus_x = rz(-np.pi/2)*gates['pi']*rz(-np.pi/2).dag()

        pi_minus_y = qt.sigmaz()*gates['pi']*qt.sigmaz()
        # do a - pi y gate on 2nd the electron
        pi_minus_oper = qt.tensor(Id2, pi_minus_y, Id2, IdN, IdN, IdN, IdN)
        # do a +pi y gate on 2nd the electron
        pi_plus_oper = qt.tensor(Id2, pi_plus_x, Id2, IdN, IdN, IdN, IdN)

        # reflect bin 0
        rho_1 = bs_QUBE[0]*(qt.tensor(rho, qt.fock_dm(N, 0)))*bs_QUBE[0].dag()
        rho_2 = (bs_QUBE[1]*(qt.tensor(rho_1, qt.fock_dm(N, 0)))*bs_QUBE[1].dag()).ptrace([0, 1, 2, 4, 5, 6, 7])
        
        # +pi around y
        rho_3 = pi_plus_oper*rho_2*pi_plus_oper.dag()
        # print('number of photons per timebin and qubit after reflect 0 =', (Noperator*rho_3.ptrace([3])).tr() + (Noperator*rho_3.ptrace([4])).tr() + (Noperator*rho_3.ptrace([5])).tr() + (Noperator*rho_3.ptrace([6])).tr())

        # reflect bin 1
        rho_4 = bs_QUBE[0]*(qt.tensor(rho_3, qt.fock_dm(N, 0)))*bs_QUBE[0].dag()
        rho_5 = (bs_QUBE[1]*(qt.tensor(rho_4, qt.fock_dm(N, 0)))*bs_QUBE[1].dag()).ptrace([0, 1, 2, 4, 5, 6, 7])
        # print('number of photons per timebin and qubit after reflect 1  =', (Noperator*rho_5.ptrace([3])).tr() + (Noperator*rho_5.ptrace([4])).tr() + (Noperator*rho_5.ptrace([5])).tr() + (Noperator*rho_5.ptrace([6])).tr())

        # -pi around y
        rho_6 = pi_plus_oper*rho_5*pi_plus_oper.dag()

        # reflect bin 2
        rho_7 = bs_QUBE[0]*(qt.tensor(rho_6, qt.fock_dm(N, 0)))*bs_QUBE[0].dag()
        rho_8 = (bs_QUBE[1]*(qt.tensor(rho_7, qt.fock_dm(N, 0)))*bs_QUBE[1].dag()).ptrace([0, 1, 2, 4, 5, 6, 7])
        # print('number of photons per timebin and qubit after reflect 2  =', (Noperator*rho_8.ptrace([3])).tr() + (Noperator*rho_8.ptrace([4])).tr() + (Noperator*rho_8.ptrace([5])).tr() + (Noperator*rho_8.ptrace([6])).tr())

        # pi around y
        rho_9 = pi_plus_oper*rho_8*pi_plus_oper.dag()

        # reflect bin 3
        rho_10 = bs_QUBE[0]*(qt.tensor(rho_9, qt.fock_dm(N, 0)))*bs_QUBE[0].dag()
        rho_11 = (bs_QUBE[1]*(qt.tensor(rho_10, qt.fock_dm(N, 0)))*bs_QUBE[1].dag()).ptrace([0, 1, 2, 4, 5, 6, 7])

        # # - pi around y
        # rho_12 = pi_minus_oper*rho_11*pi_minus_oper.dag()
        return rho_11
    
    def two_qubit_internode_blind_gate(self, een_initial, imperfections, Entange, mu):
        """ Two qubit internode blind gate """

        if imperfections['contrast_noise'] == False:
            wl_1 = self.fiber_network.siv1.optimum_freq
            wl_2 = self.fiber_network.siv2.optimum_freq
            cav_refl_1 = self.fiber_network.siv1.cav_refl(wl_1)
            cav_refl_2 = self.fiber_network.siv2.cav_refl(wl_2)
                
        elif imperfections['contrast_noise'] == True:
            wl_1 = np.random.normal(loc=self.fiber_network.siv1.optimum_freq, scale=50)
            wl_2 = np.random.normal(loc=self.fiber_network.siv2.optimum_freq, scale=50)
            cav_refl_1 = self.fiber_network.siv1.cav_refl(wl_1)
            cav_refl_2 = self.fiber_network.siv1.cav_refl(wl_2)

        # I will assume the microwave fidelities are the same in both nodes
        fidel_values_pi_pi2 = {'pi': imperfections['mw_fid_num'][0],
                    'pi_half': imperfections['mw_fid_num'][1]
                    }        
        
        # 'real'/'perfect' and 'stable'/'noisy'
        gates = set_mw_fidelities(fid = imperfections['mw'], noise = imperfections['mw_noise'], fidel_val = fidel_values_pi_pi2)

        ## first interaction of the qudit with the electron in server 1
        # print('before el1 qudit', een_initial)
        rho = self.el1_qudit_gate_QUBE(een_initial, gates, cav_refl_1, imperfections['contrast'], mu)
        # print('after el1 qudit', rho.ptrace([0, 2]))

        ## photon loss between server 
        # print(g12_b16_network.link_efficiency)
        rho2 = self.loss_photonqudit_el1el2n2(rho, self.fiber_network.link_efficiency)
        # print('number of photons per timebin and qubit after detection loss  =', (Noperator*rho2.ptrace([3])).tr() + (Noperator*rho2.ptrace([4])).tr() + (Noperator*rho2.ptrace([5])).tr() + (Noperator*rho2.ptrace([6])).tr())
        # print('after loss between links', rho2.ptrace([0, 2]))

        ## second interaction of the qudit with the elctron in server 2
        rho3 = self.el2_qudit_gate_QUBE(rho2, gates, cav_refl_2, imperfections['contrast'], mu)

        # print('after el2 qudit interaction', rho3.ptrace([0, 2]))

        ## control on the si29 pi around y on the e2
        piy = gates['pi']
        pix = rz(-np.pi/2)*gates['pi']*rz(-np.pi/2).dag()
        # now add condition to our imperfect rotation (choose mw1 or mw2 depednign on what is the right condition)
        cond_pix = cond_mw_gates(pix)['pi_mw2']
        pi_plus_y_cond_oper = qt.tensor(Id2, cond_pix, IdN, IdN, IdN, IdN)
        rho4 = pi_plus_y_cond_oper*rho3*pi_plus_y_cond_oper.dag()
        
        # print('after cond pi qudit interaction', rho4.ptrace([0, 2]))

        # qudit photon loss to detector
        # print( g12_b16_network.detection_eff)
        rho5 = self.loss_photonqudit_el1el2n2(rho4, self.fiber_network.detection_eff) #photon_loss_qudit(rho, g12_b16_network.link_efficiency)

        # print('number of photons per timebin and qubit after detection loss  =', (Noperator*rho5.ptrace([3])).tr() + (Noperator*rho5.ptrace([4])).tr() + (Noperator*rho5.ptrace([5])).tr() + (Noperator*rho5.ptrace([6])).tr())

        # measure the el2 (ancilla) in y basis
        rho6 = self.measure_el2_QUBE(rho5)

        # print('after measure ancilla interaction', rho6[0].ptrace([0, 1]))

        # measure the qudit using TDI number, the choic is based on Entange - entanglement choice, which chooses between long and short TDI
        rho7 = self.measure_qudit_QUBE(rho6[0], Entange, imperfections['tdinoise'])

        # print('measure quqit', rho7[0].ptrace([0, 1]))

        # measurement outputs s1 (0 or 1st bins), s2 (first 2 or last 2 time bins), p (ancilla output)
        measurement_outputs =  [rho7[1], rho7[2], rho6[1]]

        # add a correction to the final [el1, n2] state based on three numbers of measurements 
        rho8 = self.correction_QUBE(rho7[0], Entange, measurement_outputs)

        return rho7[0], rho8, measurement_outputs

    def measure_qudit_QUBE(self, rho, Entange, tdi_noise = 0):

        # loss and reflection for a qudit
        a_1_2 = qt.tensor(qt.destroy(N), IdN, IdN, IdN)
        a_2_2 = qt.tensor(IdN, qt.destroy(N), IdN, IdN)
        a_3_4 = qt.tensor(IdN, IdN, qt.destroy(N), IdN)
        a_4_4 = qt.tensor(IdN, IdN, IdN, qt.destroy(N))

        ## transmission and scattering beamsplitter
        a_1_3 = qt.tensor(qt.destroy(N), IdN, IdN, IdN)
        a_3_3 = qt.tensor(IdN, IdN, qt.destroy(N), IdN)
        a_2_4 = qt.tensor(IdN, qt.destroy(N), IdN, IdN)
        a_4_4 = qt.tensor(IdN, IdN, IdN, qt.destroy(N))

        Pj_1000 = qt.composite(Id2, Id2, qt.ket2dm(qt.basis(N, 1)), qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 0)))
        Pj_0100 = qt.composite(Id2, Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 1)), qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 0)))        
        Pj_0010 = qt.composite(Id2, Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 1)), qt.ket2dm(qt.basis(N, 0)))
        Pj_0001 = qt.composite(Id2, Id2, qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 0)), qt.ket2dm(qt.basis(N, 1)))        
        
        ratio = np.random.normal(loc=0.5, scale=0*0.1*0.5)
        # pick which TDI to use 
        if Entange == 1:
            # short tdi entangles
            angle = np.random.normal(loc=2*np.pi + tdi_noise, scale=0*0.1*2*np.pi)
            r = np.exp(1j*(angle))*np.sqrt(ratio)
            if np.abs(r) > 1:
                r = 1
            bs_5050_el_r_01 = general_BS(r, np.sqrt(1-(abs(r))**2), a_1_2, a_2_2)
            bs_5050_el_r_02 = general_BS(r, np.sqrt(1-(abs(r))**2), a_3_4, a_4_4)
            oper_1 = qt.tensor(Id2, Id2, bs_5050_el_r_01)
            oper_2 = qt.tensor(Id2, Id2, bs_5050_el_r_02)
            ## now the qudit is rotated into the basis we want to measure
            rho1 = oper_1*rho*oper_1.dag()
            rho2 = oper_2*rho1*oper_2.dag()
            
        elif Entange == 0:
            # long tdi doesn't entangle
            angle = np.random.normal(loc=2*np.pi + tdi_noise, scale=0*0.1*2*np.pi)
            r = np.exp(1j*(angle + np.pi/2))*np.sqrt(ratio)
            if np.abs(r) > 1:
                r = 1
            bs_5050_el_r_11 = general_BS(r, np.sqrt(1-(abs(r))**2), a_1_3, a_3_3)
            bs_5050_el_r_12 = general_BS(r, np.sqrt(1-(abs(r))**2), a_2_4, a_4_4)
            oper_1 = qt.tensor(Id2, Id2, bs_5050_el_r_11)
            oper_2 = qt.tensor(Id2, Id2, bs_5050_el_r_12)
            
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
        rho_final_bin_1 = ((Pj_1000*rho2*Pj_1000.dag())/brate_bin_1).ptrace([0, 1]) # spin state left over after bin 1 firing
        rho_final_bin_2 = ((Pj_0100*rho2*Pj_0100.dag())/brate_bin_2).ptrace([0, 1]) # spin state left over after bin 2 firing
        rho_final_bin_3 = ((Pj_0010*rho2*Pj_0010.dag())/brate_bin_3).ptrace([0, 1]) # spin state left over after bin 3 firing
        rho_final_bin_4 = ((Pj_0001*rho2*Pj_0001.dag())/brate_bin_4).ptrace([0, 1]) # spin state left over after bin 4 firing
    
        # probabilistic projective measurement
        quantum_measurement_s1s2 = np.random.choice([1, 2, 3, 4], p=[brate_bin_1_norm, brate_bin_2_norm, brate_bin_3_norm, brate_bin_4_norm])

        if Entange == 1:
            if quantum_measurement_s1s2 == 1:
                spin_state = rho_final_bin_1
                s1 = 0
                s2 = 0
            elif quantum_measurement_s1s2 == 2:
                spin_state = rho_final_bin_2
                s1 = 1
                s2 = 0
            elif quantum_measurement_s1s2 == 3:
                spin_state = rho_final_bin_3
                s1 = 0
                s2 = 1
            elif quantum_measurement_s1s2 == 4:
                spin_state = rho_final_bin_4
                s1 = 1
                s2 = 1
        elif Entange == 0:
            if quantum_measurement_s1s2 == 1:
                spin_state = rho_final_bin_1
                s1 = 0
                s2 = 0
            elif quantum_measurement_s1s2 == 2:
                spin_state = rho_final_bin_2
                s1 = 0
                s2 = 1
            elif quantum_measurement_s1s2 == 3:
                spin_state = rho_final_bin_3
                s1 = 1
                s2 = 0
            elif quantum_measurement_s1s2 == 4:
                spin_state = rho_final_bin_4
                s1 = 1
                s2 = 1
        return spin_state, s1, s2, brate_bin_1, brate_bin_2, brate_bin_3, brate_bin_4

    def correction_QUBE(self, rho, Entange, measurements):
        s1 = measurements[0]
        s2 = measurements[1]
        p = measurements[2]

        if Entange == 0:
            oper_corr = qt.tensor(s_gate()*qt.sigmaz()**(s1 + s2), (qt.sigmaz()**(p + s2)))

        elif Entange == 1:
            oper_corr = qt.tensor(s_gate()*qt.sigmaz()**(s1 + s2 + p),(qt.sigmaz()**(p + s2)))

        rho1 = oper_corr*rho*oper_corr.dag()
        return rho1

    def loss_photonqudit_el1el2n2(self, rho, eff):
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

    def measure_el2_QUBE(self, rho):
        #measure el2 in y basis

        #define a projection operator for el2
        Pj_up = qt.composite(Id2, rho_ideal_Yp, Id2, IdN, IdN, IdN, IdN) 
        Pj_down = qt.composite(Id2, rho_ideal_Ym, Id2, IdN, IdN, IdN, IdN) 

        brate_p1 = (Pj_up*rho*Pj_up.dag()).tr()
        brate_p2 = (Pj_down*rho*Pj_down.dag()).tr()
        rho_final_p1 = ((Pj_up*rho*Pj_up.dag())/brate_p1).ptrace([0, 2, 3, 4, 5, 6]) # spin state left over after apd 1
        rho_final_p2 = ((Pj_down*rho*Pj_down.dag())/brate_p2).ptrace([0, 2, 3, 4, 5, 6]) # spin state left over after apd2

        # probability of each ancilla state
        pnorm_rates = brate_p1 + brate_p2
        bprob_p1 = brate_p1 / pnorm_rates # probability of p = 0
        bprob_p2 = brate_p2 / pnorm_rates # probability of p = 1
        if np.abs(1 - (bprob_p1 + bprob_p2)) < 0.001: 
            pass   #this is in case trace above yields an approximation, in which case probs wont sum to 1 which yields error at choice
        else:
            return "Error: probabilities of p = 0 and 1 do not sum to 1"
        # probabilistic projective measurement
        quantum_measurement = np.random.choice([1,2], p=[bprob_p1, bprob_p2])
        if quantum_measurement == 1:
            rho_no_ancilla = rho_final_p1
        elif quantum_measurement == 2:
            rho_no_ancilla = rho_final_p2

        return rho_no_ancilla, quantum_measurement - 1, brate_p1, brate_p2, brate_p1 + brate_p2
    
    def el1_qudit_gate_QUBE(self, rho, gates, cav_refl, imperfections_contrast, mu):
        """ Electron QUDIT Entaglement """
        alpha = np.sqrt(mu)
        
        input_coh = generate_qudit(alpha)
        rho_0 = qt.tensor(rho, qt.ket2dm(input_coh))

        # print('Initial number of photons per timebin and qubit =', (Noperator*rho_0.ptrace([3])).tr() + (Noperator*rho_0.ptrace([4])).tr() + (Noperator*rho_0.ptrace([5])).tr() + (Noperator*rho_0.ptrace([6])).tr())
        
        bs_QUBE = siv_beamsplitter_el1_qudit_QUBE(cav_refl, imperfections_contrast)

        # print(bs_QUBE)

        # reflect bin 0
        rho_1 = (bs_QUBE[0]*(qt.tensor(rho_0, qt.ket2dm(qt.coherent(N, 0))))*bs_QUBE[0].dag())
        rho_2 = (bs_QUBE[1]*(qt.tensor(rho_1, qt.ket2dm(qt.coherent(N, 0))))*bs_QUBE[1].dag()).ptrace([0, 1, 2, 4, 5, 6, 7])

        # do a +pi y gate on the electron
        pi_plus_y = gates['pi']
        pi_plus_x = rz(-np.pi/2)*gates['pi']*rz(-np.pi/2).dag()
        pi_minus_y = qt.sigmaz()*gates['pi']*qt.sigmaz()
        pi_plus_oper = qt.tensor(pi_plus_x, Id2, Id2, IdN, IdN, IdN, IdN)
        rho_3 = pi_plus_oper*rho_2*pi_plus_oper.dag()
        # print('number of photons per timebin and qubit after reflect bin 0 =', (Noperator*rho_3.ptrace([3])).tr() + (Noperator*rho_3.ptrace([4])).tr() + (Noperator*rho_3.ptrace([5])).tr() + (Noperator*rho_3.ptrace([6])).tr())

        # reflect bin 1
        rho_4 = bs_QUBE[0]*(qt.tensor(rho_3, qt.ket2dm(qt.coherent(N, 0))))*bs_QUBE[0].dag()
        rho_5 = (bs_QUBE[1]*(qt.tensor(rho_4, qt.ket2dm(qt.coherent(N, 0))))*bs_QUBE[1].dag()).ptrace([0, 1, 2, 4, 5, 6, 7])

        # print('number of photons per timebin and qubit after reflect 1  =', (Noperator*rho_5.ptrace([3])).tr() + (Noperator*rho_5.ptrace([4])).tr() + (Noperator*rho_5.ptrace([5])).tr() + (Noperator*rho_5.ptrace([6])).tr())

        
        # reflect bin 2
        rho_6 = bs_QUBE[0]*(qt.tensor(rho_5, qt.ket2dm(qt.coherent(N, 0))))*bs_QUBE[0].dag()
        rho_7 = (bs_QUBE[1]*(qt.tensor(rho_6, qt.ket2dm(qt.coherent(N, 0))))*bs_QUBE[1].dag()).ptrace([0, 1, 2, 4, 5, 6, 7])

        # do a - pi y gate on the electron
        # pi_minus_oper = qt.tensor(pi_minus_y, Id2, Id2, IdN, IdN, IdN, IdN)
        rho_8 = pi_plus_oper*rho_7*pi_plus_oper.dag()
        
        # print('number of photons per timebin and qubit after reflect 2  =', (Noperator*rho_8.ptrace([3])).tr() + (Noperator*rho_8.ptrace([4])).tr() + (Noperator*rho_8.ptrace([5])).tr() + (Noperator*rho_8.ptrace([6])).tr())

        # reflect bin 3
        rho_9 = bs_QUBE[0]*(qt.tensor(rho_8, qt.ket2dm(qt.coherent(N, 0))))*bs_QUBE[0].dag()
        rho_10 = (bs_QUBE[1]*(qt.tensor(rho_9, qt.ket2dm(qt.coherent(N, 0))))*bs_QUBE[1].dag()).ptrace([0, 1, 2, 4, 5, 6, 7])

        # print('number of photons per timebin and qubit after reflect 3  =', (Noperator*rho_10.ptrace([3])).tr() + (Noperator*rho_10.ptrace([4])).tr() + (Noperator*rho_10.ptrace([5])).tr() + (Noperator*rho_10.ptrace([6])).tr())

        return rho_10

############## Extra Functions ###############

def siv_beamsplitter_el1_qudit_QUBE(cav_refl, contrast):
    """ Electron Photon Entaglement beamsplitter with Si29 """

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

    # loss and reflection for a qudit
    a_1_5 = qt.tensor(qt.destroy(N), IdN, IdN, IdN, IdN)
    a_5_5 = qt.tensor(IdN, IdN, IdN, IdN, qt.destroy(N))

    ## transmission and scattering beamsplitter
    a_1_6 = qt.tensor(qt.destroy(N), IdN, IdN, IdN, IdN, IdN)
    a_6_6 = qt.tensor(IdN, IdN, IdN, IdN, IdN , qt.destroy(N))
    
    ## EARLY BIN INTERACTS WITH THE SIV
    theta1_up = np.arccos(np.sqrt(1 - abs(r1_up)**2))
    phase1_up_1 = 0 - np.angle(r1_up)
    phase1_up_2 = 0 + np.angle(r1_up)

    theta1_down = np.arccos(np.sqrt(1 - abs(r1_down)**2))
    phase1_down_1 = 0 - np.angle(r1_down)
    phase1_down_2 = 0 + np.angle(r1_down)

    # Early Reflection and loss channels
    bs1_up = (((-1j*phase1_up_1/2)*(a_1_5.dag()*a_1_5 - a_5_5.dag()*a_5_5)).expm()*\
                (-theta1_up*(a_1_5.dag()*a_5_5 - a_5_5.dag()*a_1_5)).expm()*\
                ((-1j*phase1_up_2/2)*(a_1_5.dag()*a_1_5 - a_5_5.dag()*a_5_5)).expm())

    bs1_down = (((-1j*phase1_down_1/2)*(a_1_5.dag()*a_1_5 - a_5_5.dag()*a_5_5)).expm()*\
                (-theta1_down*(a_1_5.dag()*a_5_5 - a_5_5.dag()*a_1_5)).expm()*\
                ((-1j*phase1_down_2/2)*(a_1_5.dag()*a_1_5 - a_5_5.dag()*a_5_5)).expm())

    bs2_up = general_BS(sc, transm, a_1_6, a_6_6)
    bs2_down = general_BS(nsc, ntransm, a_1_6, a_6_6)

    # bs1_up = bs1_up/bs1_up.tr()
    # bs1_down = bs1_down/bs1_down.tr()
    # bs2_up = bs2_up/bs2_up.tr()
    # bs2_down = bs2_down/bs2_down.tr()

    #operation to reflect early bin
    oper1 = qt.tensor((1)*qt.ket2dm(qt.basis(2, 0)), Id2, Id2, bs1_up) + qt.tensor(((1))*qt.ket2dm(qt.basis(2, 1)), Id2, Id2, bs1_down)
    #operation to split transmission and scattering for early bin
    oper2 = qt.tensor(((1))*qt.ket2dm(qt.basis(2, 0)), Id2, Id2, bs2_up) + qt.tensor(((1))*qt.ket2dm(qt.basis(2, 1)), Id2, Id2, bs2_down)
    return oper1, oper2

def siv_beamsplitter_el2_qudit_QUBE(cav_refl, contrast):
    """ Electron 2 qudit Photon Entaglement beamsplitter with Si29 """

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

    # loss and reflection for a qudit
    a_1_5 = qt.tensor(qt.destroy(N), IdN, IdN, IdN, IdN)
    a_5_5 = qt.tensor(IdN, IdN, IdN, IdN, qt.destroy(N))

    ## transmission and scattering beamsplitter
    a_1_6 = qt.tensor(qt.destroy(N), IdN, IdN, IdN, IdN, IdN)
    a_6_6 = qt.tensor(IdN, IdN, IdN, IdN, IdN , qt.destroy(N))
    
    ## EARLY BIN INTERACTS WITH THE SIV
    theta1_up = np.arccos(np.sqrt(1 - abs(r1_up)**2))
    phase1_up_1 = 0 - np.angle(r1_up)
    phase1_up_2 = 0 + np.angle(r1_up)

    theta1_down = np.arccos(np.sqrt(1 - abs(r1_down)**2))
    phase1_down_1 = 0 - np.angle(r1_down)
    phase1_down_2 = 0 + np.angle(r1_down)

    # Early Reflection and loss channels
    bs1_up = (((-1j*phase1_up_1/2)*(a_1_5.dag()*a_1_5 - a_5_5.dag()*a_5_5)).expm()*\
                (-theta1_up*(a_1_5.dag()*a_5_5 - a_5_5.dag()*a_1_5)).expm()*\
                ((-1j*phase1_up_2/2)*(a_1_5.dag()*a_1_5 - a_5_5.dag()*a_5_5)).expm())

    bs1_down = (((-1j*phase1_down_1/2)*(a_1_5.dag()*a_1_5 - a_5_5.dag()*a_5_5)).expm()*\
                (-theta1_down*(a_1_5.dag()*a_5_5 - a_5_5.dag()*a_1_5)).expm()*\
                ((-1j*phase1_down_2/2)*(a_1_5.dag()*a_1_5 - a_5_5.dag()*a_5_5)).expm())

    bs2_up = general_BS(sc, transm, a_1_6, a_6_6)
    bs2_down = general_BS(nsc, ntransm, a_1_6, a_6_6)

    #operation to reflect early bin
    oper1 = qt.tensor(Id2, qt.ket2dm(qt.basis(2, 0)), Id2, bs1_up) + qt.tensor(Id2, qt.ket2dm(qt.basis(2, 1)), Id2, bs1_down)
    #operation to split transmission and scattering for early bin
    oper2 = qt.tensor(Id2, qt.ket2dm(qt.basis(2, 0)), Id2, bs2_up) + qt.tensor(Id2, qt.ket2dm(qt.basis(2, 1)), Id2, bs2_down)
    return oper1, oper2


 #calculate blindess of a single qubit rotation

def calculate_bloch_components(density_matrix):
    """
    Calculate the Bloch vector components (nx, ny, nz) for a given 1-qubit density matrix.

    Parameters:
    - density_matrix: A QuTiP Qobj representing a 1-qubit density matrix.

    Returns:
    - (nx, ny, nz): The components of the Bloch vector.
    """
    # Define the Pauli matrices using QuTiP
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()
    
    # Calculate the components of the Bloch vector
    nx = (density_matrix * X).tr().real  # Tr(rho * X)
    ny = (density_matrix * Y).tr().real  # Tr(rho * Y)
    nz = (density_matrix * Z).tr().real  # Tr(rho * Z)

    return nx, ny, nz

def eigenvalue_uncertainty(bloch_vector, bloch_vector_std):
    """
    Calculate the standard deviation of the eigenvalues λ± of a 1-qubit density matrix.

    Parameters:
    - bloch_vector: A tuple, list, or NumPy array of the Bloch vector components (nx, ny, nz).
    - bloch_vector_std: A tuple, list, or NumPy array of the standard deviations of the Bloch vector components (sigma_nx, sigma_ny, sigma_nz).

    Returns:
    - sigma_lambda: Standard deviation of the eigenvalues λ±.
    """
    # Unpack the grouped components
    nx, ny, nz = bloch_vector
    sigma_nx, sigma_ny, sigma_nz = bloch_vector_std
    
    # Compute the magnitude of the Bloch vector
    bloch_magnitude = np.sqrt(nx**2 + ny**2 + nz**2)
    
    # Compute the numerator of the uncertainty formula
    numerator = np.sqrt((nx * sigma_nx)**2 + (ny * sigma_ny)**2 + (nz * sigma_nz)**2)
    
    # Compute the standard deviation of the eigenvalues
    sigma_lambda = numerator / (2 * bloch_magnitude)
    
    return sigma_lambda

def entropy_uncertainty_1q(lambdas, sigma_lambdas):

    """
    Calculate the uncertainty in the von Neumann entropy S(ρ) given the uncertainties in the eigenvalues.

    Parameters:
    - lambdas: A tuple, list, or array of the eigenvalues (lambda_+, lambda_-).
    - sigma_lambdas: A tuple, list, or array of the standard deviations of the eigenvalues (sigma_lambda_+, sigma_lambda_-).

    Returns:
    - sigma_S: The uncertainty in the von Neumann entropy S(ρ).
    """
    lambda_plus, lambda_minus = lambdas
    sigma_lambda_plus, sigma_lambda_minus = sigma_lambdas
    
    # Calculate the partial derivatives
    safety = 10**(-18)
    dS_dlambda_plus = -np.log(lambda_plus + safety) - 1
    dS_dlambda_minus = -np.log(lambda_minus + safety) - 1

    # Calculate the uncertainty in S(ρ)
    sigma_S = np.sqrt((dS_dlambda_plus * sigma_lambda_plus)**2 + (dS_dlambda_minus * sigma_lambda_minus)**2)

    return sigma_S

def entropy_uncertainty_2q(lambdas, sigma_lambdas):
    """
    Calculate the uncertainty in the von Neumann entropy S(ρ) for a 2-qubit system 
    given the uncertainties in the eigenvalues.

    Parameters:
    - lambdas: A tuple, list, or array of the eigenvalues (lambda_1, lambda_2, lambda_3, lambda_4).
    - sigma_lambdas: A tuple, list, or array of the standard deviations of the eigenvalues (sigma_lambda_1, sigma_lambda_2, sigma_lambda_3, sigma_lambda_4).

    Returns:
    - sigma_S: The uncertainty in the von Neumann entropy S(ρ).
    """
    lambda_1, lambda_2, lambda_3, lambda_4 = lambdas
    sigma_lambda_1, sigma_lambda_2, sigma_lambda_3, sigma_lambda_4 = sigma_lambdas
    
    # Calculate the partial derivatives
    safety = 10**(-18)
    dS_dlambda_1 = -np.log(lambda_1 + safety) - 1
    dS_dlambda_2 = -np.log(lambda_2 + safety) - 1
    dS_dlambda_3 = -np.log(lambda_3 + safety) - 1
    dS_dlambda_4 = -np.log(lambda_4 + safety) - 1

    # Calculate the uncertainty in S(ρ)
    sigma_S = np.sqrt((dS_dlambda_1 * sigma_lambda_1)**2 + 
                      (dS_dlambda_2 * sigma_lambda_2)**2 + 
                      (dS_dlambda_3 * sigma_lambda_3)**2 + 
                      (dS_dlambda_4 * sigma_lambda_4)**2)

    return sigma_S

def holevo_bound_uncertainty_1q(rho_tot_lambdas, rho_tot_sigma_lambdas, rho_lambdas, rho_sigma_lambdas):
    """
    Calculate the uncertainty in the Holevo bound H.

    Parameters:
    - rho_tot_lambdas: Eigenvalues of rho_tot.
    - rho_tot_sigma_lambdas: Standard deviations of eigenvalues of rho_tot.
    - rho_lambdas: List of tuples (for single qubit gate) containing eigenvalues of rho1, rho2, rho3, rho4.
    - rho_sigma_lambdas: List of tuples containing standard deviations of eigenvalues of rho1, rho2, rho3, rho4.

    Returns:
    - sigma_H: The uncertainty in the Holevo bound.
    """
    # Calculate the uncertainty in S(rho_tot)
    sigma_S_tot = entropy_uncertainty_1q(rho_tot_lambdas, rho_tot_sigma_lambdas)

    # Calculate uncertainties in S(rho1), S(rho2), S(rho3), S(rho4) or as many as you have
    sigma_S_rho = [entropy_uncertainty_1q(rho_lambdas[i], rho_sigma_lambdas[i]) for i in range(len(rho_lambdas))]

    # Calculate the uncertainty in the Holevo bound
    sigma_H = np.sqrt(sigma_S_tot**2 + (1/16) * sum(s**2 for s in sigma_S_rho))
    
    return sigma_H

def holevo_bound_uncertainty_2q(rho_tot_lambdas, rho_tot_sigma_lambdas, rho_lambdas, rho_sigma_lambdas):
    """
    Calculate the uncertainty in the Holevo bound H for a 2-qubit system.

    Parameters:
    - rho_tot_lambdas: Eigenvalues of rho_tot (4 eigenvalues for 2-qubit system).
    - rho_tot_sigma_lambdas: Standard deviations of eigenvalues of rho_tot.
    - rho_lambdas: List of tuples containing eigenvalues of rho1, rho2, rho3, rho4 (each tuple has 4 eigenvalues).
    - rho_sigma_lambdas: List of tuples containing standard deviations of eigenvalues of rho1, rho2, rho3, rho4.

    Returns:
    - sigma_H: The uncertainty in the Holevo bound.
    """
    # Calculate the uncertainty in S(rho_tot) using the 2-qubit entropy uncertainty function
    sigma_S_tot = entropy_uncertainty_2q(rho_tot_lambdas, rho_tot_sigma_lambdas)

    # Calculate uncertainties in S(rho1), S(rho2), S(rho3), S(rho4) or as many as you have
    sigma_S_rho = [entropy_uncertainty_2q(rho_lambdas[i], rho_sigma_lambdas[i]) for i in range(len(rho_lambdas))]

    # Calculate the uncertainty in the Holevo bound
    sigma_H = np.sqrt(sigma_S_tot**2 + (1/16) * sum(s**2 for s in sigma_S_rho))
    
    return sigma_H

def plot_from_rho_magic(rho, title, filename, color):

    plt.rcParams.update({'font.size': 20, 'axes.linewidth': 1})


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=-70, elev=15)
    ax.set_proj_type('ortho')

    xedges = np.array([0, 1, 2])
    yedges = np.array([0, 1, 2])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.75 * np.ones_like(zpos)

    hist = np.zeros([2,2])

    for ii in range(2):
        for jj in range(2):
            hist[ii,jj] = np.abs(rho[ii,jj])

    dz = hist.ravel()

    cmap = plt.cm.get_cmap(color) # Get desired colormap - you can change this!
    
    target = (116/255, 48/255, 98/255)
    start = (1,1,1)
    diff = np.array([1-116/255, 1-48/255, 1-98/255])
    diff = diff/np.max(diff)

    N = 100
    color_list = []
    bounds = np.linspace(0,1,N)

    for ii in range(N):
        color_list.append((start[0] - diff[0]*ii/N, start[1] - diff[1]*ii/N, start[2] - diff[2]*ii/N))

    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


    max_height = 1  # get range of colorbars so we can normalize
    min_height = 0
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', linestyle='-', linewidth=1, edgecolor='k', color=rgba, shade=False)


    xedges = np.array([0, 1, 2])
    yedges = np.array([0, 1, 2])

    hist = np.array([[0, 0],
                     [0, 1]])

    zpos_hist = np.abs(np.real(rho[:]))
    dz_hist = hist - np.abs(np.real(rho[:]))

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = zpos_hist.ravel()

    # Construct arrays with the dimensions for the 4 bars.
    dx = dy = 0.75 * np.ones_like(zpos)
    dz = hist.ravel()

    hist = np.array([[0, 0],
                     [0, 1]])

    dz = dz_hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1, edgecolor='grey', color=(0, 0, 1, 0))


    ax.plot([0.05,0.05],[0.05,0.05],[-0.01,1], color='k', linewidth=1)
    #ax.plot([0.0,0.0],[2.3,4.3],[0,0.5], color='k', linewidth=1.5)


    ax.set_zticks([0, 0.5, 1])
    ax.set_zlim([0,1])
    ax.set_zticklabels(["0", "", "1"])

    ax.set_xticks([0.5, 1.5])
    ax.set_xlim([0.09,2.2])
    ax.set_xticklabels(["-TXT", "+TXT"])
    ax.set_yticks([0.5, 1.5])
    ax.set_ylim([0.09,2.2])
    ax.set_yticklabels(["-TXT", "+TXT"])

    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.edge = 0

    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
    ax.zaxis._axinfo["grid"]['linewidth'] =  1
    ax.zaxis._axinfo["tick"]['lenght'] =  0

    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(color=(0,0,0,0))

    ax.w_zaxis.linewidth =  1
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0", "1"])

    ax.set_title(title)


    plt.show()
    fig.savefig(filename, bbox_inches='tight', dpi=300)

def plot_from_rho_Identity(rho, title, filename, color):

    plt.rcParams.update({'font.size': 20, 'axes.linewidth': 1})
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=-70, elev=15)
    ax.set_proj_type('ortho')

    xedges = np.array([0, 1, 2])
    yedges = np.array([0, 1, 2])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.75 * np.ones_like(zpos)

    hist = np.zeros([2,2])

    for ii in range(2):
        for jj in range(2):
            hist[ii,jj] = np.abs(rho[ii,jj])
            # print(hist[ii,jj])

    dz = hist.ravel()

    cmap = plt.cm.get_cmap(color) # Get desired colormap - you can change this!
    
    target = (241/255, 95/255, 88/255)
    start = (1,1,1)
    diff = np.array([1-241/255, 1-95/255, 1-88/255])
    diff = diff/np.max(diff)

    N = 100
    color_list = []
    bounds = np.linspace(0,1,N)

    for ii in range(N):
        color_list.append((start[0] - diff[0]*ii/N, start[1] - diff[1]*ii/N, start[2] - diff[2]*ii/N))

    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


    max_height = 1  # get range of colorbars so we can normalize
    min_height = 0
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', linestyle='-', linewidth=1, edgecolor='k', color=rgba, shade=False)


    xedges = np.array([0, 1, 2])
    yedges = np.array([0, 1, 2])

    hist = np.array([[0, 0],
                     [0, 1]])

    zpos_hist = np.abs(np.real(rho[:]))
    dz_hist = hist - np.abs(np.real(rho[:]))
    # print('zpos_hist', zpos_hist)
    # print('dz_hist', dz_hist)


    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = zpos_hist.ravel()

    # Construct arrays with the dimensions for the 4 bars.
    dx = dy = 0.75 * np.ones_like(zpos)
    dz = hist.ravel()

    hist = np.array([[0, 0],
                     [0, 1]])

    dz = dz_hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1, edgecolor='grey', color=(0, 0, 1, 0))


    ax.plot([0.05,0.05],[0.05,0.05],[-0.01,1], color='k', linewidth=1)
    #ax.plot([0.0,0.0],[2.3,4.3],[0,0.5], color='k', linewidth=1.5)


    ax.set_zticks([0, 0.5, 1])
    ax.set_zlim([0,1])
    ax.set_zticklabels(["0", "", "1"])

    ax.set_xticks([0.5, 1.5])
    ax.set_xlim([0.09,2.2])
    ax.set_xticklabels(["$-Y$", "$+Y$"])

    ax.set_yticks([0.5, 1.5])
    ax.set_ylim([0.09,2.2])
    ax.set_yticklabels(["$-Y$", "$+Y$"])

    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.edge = 0

    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
    ax.zaxis._axinfo["grid"]['linewidth'] =  1
    ax.zaxis._axinfo["tick"]['lenght'] =  0

    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(color=(0,0,0,0))

    ax.w_zaxis.linewidth =  1
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0", "1"])

    ax.set_title(title)


    plt.show()
    fig.savefig(filename, bbox_inches='tight', dpi=300)

def plot_from_rho(rho, title, filename, color, ON):
    # not sensitive to imaginary elements of the density matrix
    plt.rcParams.update({'font.size': 20, 'axes.linewidth': 1.5})


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=-70, elev=25)
    ax.set_proj_type('ortho')

    xedges = np.array([0, 1, 2, 3, 4])
    yedges = np.array([0, 1, 2, 3, 4])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.75 * np.ones_like(zpos)

    hist = np.zeros([4,4])

    for ii in range(4):
        for jj in range(4):
            hist[ii,jj] = np.abs(np.abs(rho[ii,jj]))

    dz = hist.ravel()

    #cmap = plt.cm.get_cmap(color) # Get desired colormap - you can change this!
    
    target = color
    start = (1,1,1)
    diff = np.array([1-target[0], 1-target[1], 1-target[2]])
    diff = diff/np.max(diff)

    N = 100
    color_list = []
    bounds = np.linspace(0,1,N)

    for ii in range(N):
        color_list.append((start[0] - diff[0]*ii/N, start[1] - diff[1]*ii/N, start[2] - diff[2]*ii/N))

    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if ON:
        max_height = 0.5  # get range of colorbars so we can normalize
    else:
        max_height = 1.0
    min_height = 0
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    xpos_ori, ypos_ori, zpos_ori, dx_ori, dy_ori, dz_ori \
        = xpos, ypos, zpos, dx.copy(), dy.copy(), dz.copy()
#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', linestyle='-', linewidth=1.5, edgecolor='k', color=rgba, shade=False)


    xedges = np.array([0, 3, 6])
    yedges = np.array([0, 3, 6])

    if ON:
        hist = 0.5*np.ones([2,2])


        zpos_hist = np.abs(np.real(rho[0:4:3,0:4:3]))
        dz_hist = hist - np.abs(np.real(rho[0:4:3,0:4:3]))

        # Construct arrays for the anchor positions of the 16 bars.
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = zpos_hist.ravel()

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.75 * np.ones_like(zpos)
        dz = hist.ravel()

        hist = 0.5*np.ones([2,2])

        dz = dz_hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1.5, edgecolor='grey', color=(0, 0, 1, 0))


        ax.plot([0,0],[0,0],[-0.01,0.5], color='k', linewidth=1.5)
        ax.plot([0.0,0.0],[4.3,4.3],[0,0.5], color='k', linewidth=1.5)


        ax.set_zticks([0, 0.25, 0.5])
        ax.set_zlim([0,0.5])
        ax.set_zticklabels(["0", "", "0.5"])

        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xlim([0.09,4.2])
        ax.set_xticklabels(["$+i+i$", "$+i-i$", "$-i +i$", "$-i-i$"])

        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_ylim([0.09,4.2])
        ax.set_yticklabels(["$+i+i$", "$+i-i$", "$-i +i$", "$-i-i$"])

        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.edge = 0

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
        ax.zaxis._axinfo["grid"]['linewidth'] =  1.5
        ax.zaxis._axinfo["tick"]['lenght'] =  0

        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(color=(0,0,0,0))

        ax.w_zaxis.linewidth =  1.5
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1.0"])

    else:
        hist = np.array([[0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,1]])
        
        zpos_hist = np.abs(np.real(rho))
        dz_hist = hist - np.abs(np.real(rho))

        # Construct arrays for the anchor positions of the 16 bars.
#         xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos, ypos = np.meshgrid(np.array([0, 1, 2, 3]) + 0.25, np.array([0, 1, 2, 3]) + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = zpos_hist.ravel()

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.75 * np.ones_like(zpos)
        dz = hist.ravel()

        hist = 0.5*np.ones([4,4])

        dz = dz_hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min', linestyle='--', linewidth=1.5, edgecolor='grey', color=(0, 0, 1, 0))


        ax.plot([0,0],[0,0],[-0.01,1.0], color='k', linewidth=1.5)
        ax.plot([0.0,0.0],[4.3,4.3],[0,1.0], color='k', linewidth=1.5)


        ax.set_zticks([0, 0.5, 1.0])
        ax.set_zlim([0,1.0])
        ax.set_zticklabels(["0", "", "1"])

        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xlim([0.09,4.2])
        ax.set_xticklabels(["$- -$", "$- +$", "$+ -$", "$+ +$"])

        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_ylim([0.09,4.2])
        ax.set_yticklabels(["$- -$", "$- +$", "$+ -$", "$+ +$"])

        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.edge = 0

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,1)
        ax.zaxis._axinfo["grid"]['linewidth'] =  1.5
        ax.zaxis._axinfo["tick"]['lenght'] =  0

        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(color=(0,0,0,0))

        ax.w_zaxis.linewidth =  1.5
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.04, pad=0.04, aspect=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1.0"])
        
    ax.bar3d(xpos_ori, ypos_ori, zpos_ori, dx_ori, dy_ori, dz_ori, \
             zsort='average', linestyle='-', linewidth=1.5, edgecolor='k', color=rgba, shade=False)
    ax.set_title(title)


    plt.show()
    fig.savefig(filename, bbox_inches='tight', dpi=300)