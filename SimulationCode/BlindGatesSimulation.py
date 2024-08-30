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
        imperfections['mw_fid_num']
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
            eff = 0.9
            rho_2l = loss_photonqubit_elSpin(rho_2, eff)

            """Measure photon in the chosen basis"""
            rho_3 = phi_photon_measurement(rho_2l, phi1)

        """Measure electron"""
    
        fidX_c = ((-1)**rho_3[1])*round(qt.fidelity(rho_3[0], rho_ideal_Xp)**2 - qt.fidelity(rho_3[0], rho_ideal_Xm)**2, 3)
        fidY_c = ((-1)**rho_3[1])*round(qt.fidelity(rho_3[0], rho_ideal_Yp)**2 - qt.fidelity(rho_3[0], rho_ideal_Ym)**2, 3)
        fidZ_c = round(qt.fidelity(rho_3[0], rho_ideal_Zp)**2 - qt.fidelity(rho_3[0], rho_ideal_Zm)**2, 3)
 
        fidX_s = round(qt.fidelity(rho_3[0], rho_ideal_Xp)**2 - qt.fidelity(rho_3[0], rho_ideal_Xm)**2, 3)
        fidY_s = round(qt.fidelity(rho_3[0], rho_ideal_Yp)**2 - qt.fidelity(rho_3[0], rho_ideal_Ym)**2, 3)
        fidZ_s = round(qt.fidelity(rho_3[0], rho_ideal_Zp)**2 - qt.fidelity(rho_3[0], rho_ideal_Zm)**2, 3)

        return rho_3, fidX_c, fidY_c, fidZ_c, fidX_s, fidY_s, fidZ_s
    


############## Extra Functions ###############

 #calculate blindess of a single qubit rotation

# def blindness_singleRot(rho_phi_array, rho_std_phi_array, delta=1e-6):
#         """
#         Calculate the Holevo bound and its uncertainty given an array of density matrices and their standard deviations.
        
#         Parameters:
#         - rho_phi_array: List or array of density matrices corresponding to different phases.
#         - rho_std_phi_array: List or array of standard deviations for each element in the density matrices.
#         - delta: Small perturbation value for numerical derivative (default is 1e-6).
        
#         Returns:
#         - holevo: The calculated Holevo bound.
#         - holevo_error: The uncertainty in the Holevo bound.
#         """
        
#         # Calculate the average density matrix
#         rho_all = np.mean(rho_phi_array, axis=0)
        
#         # Calculate the mean entropy of the individual density matrices
#         mn = np.mean([qt.entropy_vn(qt.Qobj(rho)) for rho in rho_phi_array])

#         # Calculate the Holevo bound
#         holevo = qt.entropy_vn(qt.Qobj(rho_all)) - mn

#         # Initialize the error in the Holevo bound
#         holevo_error = 0
        
#         for k, rho in enumerate(rho_phi_array):
#             partial_derivatives = np.zeros(rho.shape, dtype=complex)
#             chi = holevo
            
#             # Calculate the partial derivatives numerically
#             for i in range(rho.shape[0]):
#                 for j in range(rho.shape[1]):
#                     perturbed_rho = rho.copy()
                    
#                     # Apply perturbation and ensure Hermiticity
#                     delta_rho = np.zeros(rho.shape, dtype=complex)
#                     delta_rho[i, j] = delta
#                     delta_rho[j, i] = np.conj(delta_rho[i, j])  # Ensure Hermiticity
                    
#                     # Adjust diagonal elements to preserve trace
#                     trace_adjustment = np.trace(delta_rho)
#                     delta_rho[i, i] -= trace_adjustment / rho.shape[0]
                    
#                     perturbed_rho += delta_rho
                    
#                     # Calculate perturbed average density matrix
#                     perturbed_rho_all = np.mean(
#                         [perturbed_rho if idx == k else rho_phi_array[idx] for idx in range(len(rho_phi_array))],
#                         axis=0
#                     )
                    
#                     # Recalculate mean entropy
#                     perturbed_mn = np.mean([qt.entropy_vn(qt.Qobj(rho_phi)) for rho_phi in rho_phi_array])
                    
#                     # Calculate perturbed Holevo bound
#                     perturbed_chi = qt.entropy_vn(qt.Qobj(perturbed_rho_all)) - perturbed_mn
                    
#                     # Calculate the partial derivative
#                     partial_derivatives[i, j] = (perturbed_chi - chi) / delta
            
#             # Sum the squared errors propagated through each partial derivative
#             holevo_error += np.sum((np.abs(partial_derivatives) * rho_std_phi_array[k]) ** 2)
        
#         # Take the square root to obtain the final uncertainty
#         holevo_error = np.abs(np.sqrt(holevo_error))
        
#         return holevo, holevo_error
