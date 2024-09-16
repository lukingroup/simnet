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
    

        if imperfections['contrast_noise'] == False:
            wl = self.fiber_network.siv1.optimum_freq
            cav_refl = self.fiber_network.siv1.cav_refl(wl)
            
        elif imperfections['contrast_noise'] == True:
            wl = np.random.normal(loc=self.fiber_network.siv1.optimum_freq, scale=50)
            cav_refl = self.fiber_network.siv1.cav_refl(wl)

        siv_beamsplitters = self.siv_beamsplitter_si29(cav_refl, imperfections['contrast'])

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
        eff = sim.fiber_network.detection_eff
        rho_2l = loss_photonqubit_elSpin_withsi29(rho_2, eff)
        
        """Measure photon in the chosen basis"""
        rho_3 = phi_photon_measurement_withsi29(rho_2l, phi, imperfections['tdinoise'])

        return rho_3
    
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
    sigma_S_rho = [entropy_uncertainty(rho_lambdas[i], rho_sigma_lambdas[i]) for i in range(len(rho_lambdas))]

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

