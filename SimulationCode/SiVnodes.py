#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 4 11:28:29 2023

@author: azizasuleymanzade
"""
import numpy as np
import qutip as qt

class SiV:
    def __init__(self, kappa_in='kappa_in', kappa_w='kappa_w', g='g', wCav = 'wCav', 
                 wSiv = 'wSiV', dwEl = 'dwEl'):
        """Initialize the SiV center with basic properties."""
        self.kappa_in = kappa_in        #internal linewidth GHz
        self.kappa_w = kappa_w          #coupling linewidth GHz
        self.g = g                      #coupling strength GHz
        self.wSiv = wSiv                #frequency of SiV/lowest electron state MHz
        self.wCav = wCav                #frequency of the cavity resonance MHz
        self.dwEl = dwEl                #optical electron splitting MHz
        self.gamma = 0.05*10**3       #bare linewidth of the SiV
        self.optimum_freq =  self.freq_optimum() # max contrast frequency
        self.optimum_refl = self.cav_refl(self.optimum_freq) # max contrast reflectivities of electron states

    # calculates complex reflectivities at frequency wl
    def cav_refl(self, wl):
        
        """Find reflectivity of the SiV states (DOUBLE CHECK TRANSMISSION AND SCATTERING)""" 
        # make each into a function

        kappa_tot = self.kappa_in + self.kappa_w

        #reflection of non-reflecting state
        nonrefl_refl = 1 - 2*self.kappa_w/(1j*(wl - self.wCav) + kappa_tot + self.g**2/(1j*(wl - self.wSiv) + self.gamma)) #reflectivity formula
        
        #transmission of non-reflecting state
        nonrefl_tr = 2*(np.sqrt(self.kappa_in*self.kappa_w))/(1j*(wl -self.wCav) \
            + kappa_tot + self.g**2/(1j*(wl - self.wSiv) + self.gamma))
        
        #scattering of non-reflecting state 
        nonrefl_sc= 2*np.sqrt(self.kappa_in*self.gamma)*self.g / ((1j*(wl - self.wCav)+\
            kappa_tot)*(1j*(wl - self.wSiv)+self.gamma) + self.g**2)
        
        #reflection of reflecting state
        refl_refl = 1 - 2*self.kappa_w/(1j*(wl - self.wCav) +\
            kappa_tot + self.g**2/(1j*(wl - (self.wSiv + self.dwEl))+ self.gamma))

        #transmission of reflecting state
        refl_tr = 2*(np.sqrt(self.kappa_in*self.kappa_w))/(1j*(wl - self.wCav) \
            + kappa_tot + self.g**2/(1j*(wl - (self.wSiv + self.dwEl)) + self.gamma))
        
        #scattering of reflecting state
        refl_sc = 2*np.sqrt(self.kappa_in*self.gamma)*self.g / ((1j*(wl - self.wCav)+\
            kappa_tot)*(1j*(wl - (self.wSiv + self.dwEl))+self.gamma) + self.g**2)
            
        dct = {'nonrefl_refl': nonrefl_refl,
               'nonrefl_tr': nonrefl_tr,
               'nonrefl_sc': nonrefl_sc,
               'refl_refl': refl_refl,
               'refl_tr': refl_tr,
               'refl_sc': refl_sc
               }
        
        return dct
    
    # calculates optimum frequency for contrast
    def freq_optimum(self):
        
        """Find the optical frequency for the highest contrast"""
        
        wl = self.get_plotaxis()
        
        refl= np.zeros(wl.shape)
        refl_phase= np.zeros(wl.shape)
        nonrefl= np.zeros(wl.shape)
        nonrefl_phase= np.zeros(wl.shape)
        
        for i in range(len(wl)):
            mag_refl = np.abs(self.cav_refl(wl[i])['refl_refl']) #magnitude of amplitude of reflection of reflecting state 
            mag_nonrefl = np.abs(self.cav_refl(wl[i])['nonrefl_refl']) #magnitude of amplitude of reflection of non-reflecting state
            #probe intensity
            refl[i]= mag_refl**2
            refl_phase[i] = np.angle(self.cav_refl(wl[i])['refl_refl']) #arg (phase) of reflection of reflecting state
            nonrefl[i]= mag_nonrefl**2 #nonrefl_A!
            nonrefl_phase[i] = np.angle(self.cav_refl(wl[i])['nonrefl_refl']) #arg (phase) of reflection of non-reflecting state
    
        contrast = refl/nonrefl
        ind = np.argmax(contrast)
        wl_read_optimum = wl[ind]
        # print(contrast)
        

        # print(contrast[ind])
        # print(wl_read_optimum)

        return wl_read_optimum
    
    def get_plotaxis(self):
        
        """Find the optical frequency for the highest contrast"""
        wl = np.arange(start=self.wSiv - 2*10**3, stop=self.wSiv + 2*10**3, step = 0.01*10**3)
        return wl
    
    # gives the value of the contrast at the optimum frequency
    def get_best_contrast(self):
        contrast = (np.abs(self.optimum_refl['refl_refl'])**2)/(np.abs(self.optimum_refl['nonrefl_refl'])**2)
        return contrast
    
    def best_wSiV(self, wsiv_start, wsiv_end):
        wsivList = np.linspace(wsiv_start, wsiv_end, int(1*10**3))
    
        nonrefl_min= np.zeros(wsivList.shape)

        for j in range(len(wsivList)):
            self.wSiv = wsivList[j]
            wl = np.arange(start=self.wSiv - 0.5*10**3, stop=self.wSiv + 0.5*10**3, step = 0.001*10**3)
            nonrefl= np.zeros(wl.shape)
            for i in range(len(wl)):
                mag_nonrefl = np.abs(self.cav_refl(wl[i])['nonrefl_refl']) #magnitude of amplitude of reflection of non-reflecting state
                nonrefl[i]= mag_nonrefl**2 #nonrefl_A!
            nonrefl_min[j] = np.min(nonrefl)
            # print("lowest refl = ", nonrefl_min[j], "at wsiv = ", self.wSiv)
        
        ind = np.argmin(nonrefl_min)
        self.wSiv = wsivList[ind]
        self.optimum_freq =  self.freq_optimum() # max contrast frequency
        self.optimum_refl = self.cav_refl(self.optimum_freq) #

        return

    # sets contrast by chenging the frequency away from optimum
    def set_contrast(self, goal_contrast):
        
        # freq = self.freq_optimum()
        det = 100
        count = 0
        improve = True
    
        contrast = self.get_best_contrast()
        # print("init sIV = ", self.wSiv )

        while(np.abs(goal_contrast - contrast)>3):  

            if (goal_contrast > contrast):
                # print('increase contrast from contrast  = ', contrast)
                self.wSiv = self.wSiv + det
                # print("move sIV = ", self.wSiv)

                self.optimum_freq =  self.freq_optimum() # max contrast frequency
                self.optimum_refl = self.cav_refl(self.optimum_freq) #
                contrast = self.get_best_contrast()
                # print('new contrast  = ', contrast)
    

            elif (goal_contrast < contrast):
                # print('decrease contrast from contrast  = ', contrast)
                self.wSiv = self.wSiv - det
                # print("move sIV = ", self.wSiv)
                self.optimum_freq =  self.freq_optimum() # max contrast frequency
                self.optimum_refl = self.cav_refl(self.optimum_freq) #
                contrast = self.get_best_contrast()
                # print('new contrast  = ', contrast)
                
            
            # print(improve)

            count += 1
    
            if count >= 2000:
                print("Breaking out of the loop.")
                break  # Exits the while loop when count reaches 5
        print("new contrast", contrast)
        return
    

# Create SiVs:
# siv_a = SiV(kappa_in= 56*(10**3), kappa_w= (74 - 56)*(10**3), g=7.8*(10**3), wCav = (0)*(10**3), 
#              wSiv = 110*(10**3), dwEl = 0.5*(10**3)) # G12
# siv_b = SiV(kappa_in= 101*(10**3), kappa_w= 29*(10**3), g=3.19*(10**3), wCav = (0)*(10**3), 
#              wSiv = 55.0*(10**3), dwEl = 0.5*(10**3)) # B16

# contrast = siv_a.get_best_contrast()

# siv_a.freq_optimum()

# print("Contrast G12 at Best contrast of A: ", contrast_at_best_A(delta))
# print("Contrast B16 at Best contrast of B : ", (np.abs(cav_refl(1 , output_optimums(delta)[1], delta)[3])**2)/(np.abs(cav_refl(1 , output_optimums(delta)[1], delta)[0])**2))
#         return contrast