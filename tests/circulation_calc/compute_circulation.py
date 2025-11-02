# FUNCTION - COMPUTE CIRCULATION

import numpy as np
from scipy import interpolate

def COMPUTE_CIRCULATION(a,b,x0,y0,numT,Vx,Vy,X,Y):
    
    t     = np.linspace(0,2*np.pi,numT)                                         # Discretized ellipse into angles [rad]
    xC    = a*np.cos(t) + x0                                                    # X coordinates of ellipse
    yC    = b*np.sin(t) + y0                                                    # Y coordinates of ellipse
    fx    = interpolate.RectBivariateSpline(Y,X,Vx)                             # Interpolate X velocities from grid to ellipse points
    fy    = interpolate.RectBivariateSpline(Y,X,Vy)                             # Interpolate Y velocities from grid to ellipse points
    VxC   = fx.ev(yC,xC)                                                        # X velocity component on ellipse
    VyC   = fy.ev(yC,xC)                                                        # Y velocity component on ellipse
    Gamma = -(np.trapz(VxC,xC) + np.trapz(VyC,yC))                              # Compute integral using trapezoid rule
    
    return Gamma, xC, yC, VxC, VyC