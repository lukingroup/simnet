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
    
    # def single_qubit_universal_blind_gate(self, el_initial, imperfections, cluster_state_length, phi_array, mu):
        
    #     # define hadamart
    #     Had = ry(-np.pi/2) # Ry for hadamard

    #     measurement_outputs = np.empty((0, 2), dtype=float)

    #     rho = el_initial
    #     for i in range(cluster_state_length): 
    #         rho_after_SPG = single_rotation_electron(self, rho, imperfections, phi_array[i], mu)
    #         rho = Had*rho_after_SPG[0]*Had.dag()
    #         measurement_outputs = np.append(measurement_outputs, [rho_after_SPG[2::]])

    #     return rho, measurement_outputs
    

############## Extra Functions ###############

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

import numpy as np

def entropy_uncertainty(lambdas, sigma_lambdas):
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

def holevo_bound_uncertainty(rho_tot_lambdas, rho_tot_sigma_lambdas, rho_lambdas, rho_sigma_lambdas):
    """
    Calculate the uncertainty in the Holevo bound H.

    Parameters:
    - rho_tot_lambdas: Eigenvalues of rho_tot.
    - rho_tot_sigma_lambdas: Standard deviations of eigenvalues of rho_tot.
    - rho_lambdas: List of tuples containing eigenvalues of rho1, rho2, rho3, rho4.
    - rho_sigma_lambdas: List of tuples containing standard deviations of eigenvalues of rho1, rho2, rho3, rho4.

    Returns:
    - sigma_H: The uncertainty in the Holevo bound.
    """
    # Calculate the uncertainty in S(rho_tot)
    sigma_S_tot = entropy_uncertainty(rho_tot_lambdas, rho_tot_sigma_lambdas)

    # Calculate uncertainties in S(rho1), S(rho2), S(rho3), S(rho4)
    sigma_S_rho = [entropy_uncertainty(rho_lambdas[i], rho_sigma_lambdas[i]) for i in range(4)]

    # Calculate the uncertainty in the Holevo bound
    sigma_H = np.sqrt(sigma_S_tot**2 + (1/16) * sum(s**2 for s in sigma_S_rho))
    
    return sigma_H

# # Example usage
# rho_tot_lambdas = (0.7, 0.3)  # Example eigenvalues of rho_tot
# rho_tot_sigma_lambdas = (0.02, 0.01)  # Example uncertainties in eigenvalues of rho_tot

# rho_lambdas = [(0.6, 0.4), (0.5, 0.5), (0.55, 0.45), (0.65, 0.35)]  # Eigenvalues of rho1, rho2, rho3, rho4
# rho_sigma_lambdas = [(0.01, 0.02), (0.02, 0.02), (0.015, 0.015), (0.01, 0.01)]  # Uncertainties in eigenvalues

# sigma_H = holevo_bound_uncertainty(rho_tot_lambdas, rho_tot_sigma_lambdas, rho_lambdas, rho_sigma_lambdas)
# print(f"Uncertainty in the Holevo bound H: {sigma_H:.6f}")



