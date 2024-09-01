#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:57:03 2023

@author: azizasuleymanzade
"""

from SiVnodes import SiV

class FiberNetwork:
    def __init__(self, siv1 = 'siv1', siv2 = 'siv2'):
        """Initialize the network with an initial state."""
        self.siv1 = siv1
        self.siv2 = siv2
        self.fibercoupling_eff  = 1
        self.tdi_eff  = 1
        self.snspd_eff  = 1
        self.aom_eff  = 1
        self.detection_eff = self.fibercoupling_eff*self.aom_eff*self.tdi_eff*self.snspd_eff

    def detection_eff_reset(self):
        self.detection_eff = self.fibercoupling_eff*self.aom_eff*self.tdi_eff*self.snspd_eff
        return self.detection_eff