import numpy as np
import math as math

np.seterr('raise')

def STREAMLINE_SPM(XP,YP,XB,YB,phi,S):
    
    # Number of panels
    numPan = len(XB)-1                                                          # Number of panels (control points)
    
    # Initialize arrays
    Mx = np.zeros(numPan)                                                       # Initialize Mx integral array
    My = np.zeros(numPan)                                                       # Initialize My integral array
    
    # Compute Mx and My
    for j in range(numPan):                                                     # Loop over all panels
        # Compute intermediate values
        A = -(XP-XB[j])*np.cos(phi[j]) - (YP-YB[j])*np.sin(phi[j])              # A term
        B  = (XP-XB[j])**2 + (YP-YB[j])**2;                                     # B term
        Cx = -np.cos(phi[j]);                                                   # Cx term (X-direction)
        Dx = XP-XB[j];                                                          # Dx term (X-direction)
        Cy = -np.sin(phi[j]);                                                   # Cy term (Y-direction)
        Dy = YP-YB[j];                                                          # Dy term (Y-direction)
        E  = math.sqrt(B-A**2);                                                 # E term
        if (E == 0 or np.iscomplex(E) or np.isnan(E) or np.isinf(E)):           # If E term is 0 or complex or a NAN or an INF
            Mx[j] = 0                                                           # Set Mx value equal to zero
            My[j] = 0                                                           # Set My value equal to zero
        else:
            # Compute Mx, Ref [1]
            term1 = 0.5*Cx*np.log((S[j]**2 + 2*A*S[j]+B)/B);                    # First term in Mx equation
            term2 = ((Dx-A*Cx)/E)*(math.atan2((S[j]+A),E) - math.atan2(A,E));   # Second term in Mx equation
            Mx[j] = term1 + term2;                                              # Compute Mx integral
            
            # Compute My, Ref [1]
            term1 = 0.5*Cy*np.log((S[j]**2 + 2*A*S[j]+B)/B);                    # First term in My equation
            term2 = ((Dy-A*Cy)/E)*(math.atan2((S[j]+A),E) - math.atan2(A,E));   # Second term in My equation
            My[j] = term1 + term2;                                              # Compute My integral

        # Zero out any problem values
        if (np.iscomplex(Mx[j]) or np.isnan(Mx[j]) or np.isinf(Mx[j])):         # If Mx term is complex or a NAN or an INF
            Mx[j] = 0                                                           # Set Mx value equal to zero
        if (np.iscomplex(My[j]) or np.isnan(My[j]) or np.isinf(My[j])):         # If My term is complex or a NAN or an INF
            My[j] = 0                                                           # Set My value equal to zero
    
    return Mx, My                                                               # Return both Mx and My matrices

