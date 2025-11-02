# FUNCTION - COMPUTE CIRCULATION

import numpy as np
from scipy import interpolate

def COMPUTE_CIRCULATION(a,b,x0,y0,numT,Vx,Vy,X,Y):
    
    # Parameterise ellipse
    t     = np.linspace(0,2*np.pi,numT)
    xC    = a*np.cos(t) + x0
    yC    = b*np.sin(t) + y0

    # Interpolate vector field onto ellipse
    fx    = interpolate.RectBivariateSpline(Y,X,Vx)
    fy    = interpolate.RectBivariateSpline(Y,X,Vy)
    VxC   = fx.ev(yC,xC)
    VyC   = fy.ev(yC,xC)

    # Circulation = -integral(VxC*dx + VyC*dy)
    Gamma = -(np.trapz(VxC,xC) + np.trapz(VyC,yC))
    
    return Gamma, xC, yC, VxC, VyC