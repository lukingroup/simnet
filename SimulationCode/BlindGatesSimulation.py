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
            # print(imperfections['contrast_noise'], 1)
        elif imperfections['contrast_noise'] == True:
            wl = np.random.normal(loc=self.fiber_network.siv1.optimum_freq, scale=50)
            cav_refl = self.fiber_network.siv1.cav_refl(wl)
            # print(imperfections['contrast_noise'], 2)

        siv_beamsplitters = siv_beamsplitter(cav_refl, imperfections['contrast'])
        # print(siv_beamsplitters)
        gates = mw_gates('perfect', 'stable')

        """Initial state"""

        rho = 0
        for i in np.arange(cluster_state_length):

            """Spin photon entanglement"""
            rho_1 = spin_photon_entaglement(siv_beamsplitters, el_initial, gates['pi'], mu)

            """ - pi on the electron, to associate up spin with early timebin"""
            oper_pi = qt.tensor(gates['pi'], IdN, IdN)
            rho_2 = oper_pi*rho_1*oper_pi.dag() 

            """Add losses if needed"""
            eff = 0.1
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
    
    def blindness_singleRot(self, rho_phi_array):
        # rho_phi_array has electron density matricies for i = 0,1,2,3 corresponding to 0, pi , pi/2 3pi/2
        N_phi = len(rho_phi_array)
        rho_all = np.mean(rho_phi_array)
        print(rho_all)
        holevo = qt.entropy_vn(rho_all) - (1/N_phi)*np.mean(qt.entropy_vn(rho_phi_array))
        # holevo_error = 
        return holevo

    def plot_SiV(self):
        """Plotting Function"""
