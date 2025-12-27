import numpy as np

def parse_naca4(code: str):
    """
    Parse a NACA 4-digit airfoil code.

    Parameters
    ----------
    code : str
        4-digit NACA code (e.g. "2412")

    Returns
    -------
    m : float
        Maximum camber as fraction of chord
    p : float
        Location of maximum camber (fraction of chord)
    t : float
        Maximum thickness as fraction of chord
    """
    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA code must be a 4-digit string, e.g. '2412'")

    A = int(code[0])
    B = int(code[1])
    C = int(code[2:4])

    m = A / 100.0
    p = B / 10.0
    t = C / 100.0

    return m, p, t
    
def gen_naca4(code: str, 
              chord : float,
              n_chord : int):
    """
    Generate the boundary coordinates of a NACA 4-digit airfoil.

    Parameters
    ----------
    code : str
        4-digit NACA code (e.g. "2412").
    chord : float
        Chord length to scale the airfoil.
    n_chord : int
        Number of points along the chord (cosine-spaced for better LE resolution).

    Returns
    -------
    numpy.ndarray
        Array of shape (2 * n_chord - 1, 2) with x, y coordinates ordered
        from the trailing edge along the upper surface to the leading edge,
        then back along the lower surface to the trailing edge.
    """
              
    m, p, t = parse_naca4(code)    

    # Apply cosine spacing for x so that
    # points are clustered near LE
    beta = np.linspace(0, np.pi, n_chord)
    x = 0.5 * (1 - np.cos(beta))

    y_c = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    #define mask for points before and after max camber
    mask = x<=p

    # define the camber line y_c, and dyc_dx
    if p != 0:
        y_c[mask] = m*(2*p*x[mask] - x[mask]**2)/p**2 
        y_c[~mask] = m*(1 - 2*p + 2*p*x[~mask] - x[~mask]**2)/(1-p)**2
        dyc_dx[mask] = 2*m*(p-x[mask])/p**2 
        dyc_dx[~mask] = 2*m*(p-x[~mask])/((1-p)**2)

    # define y_t, the thickness line
    y_t = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)

    # theta along camber = arctan(dy_c/dx)     
    theta = np.arctan(dyc_dx)

    # x_u, y_u = x - y_t*sin(theta), y_c + y_t*cos(theta)
    x_u, y_u = x - y_t*np.sin(theta), y_c + y_t*np.cos(theta)

    # x_l, y_l = x + y_t*sin(theta), y_c - y_t*cos(theta)
    x_l, y_l = x + y_t*np.sin(theta), y_c - y_t*np.cos(theta)

    x_u *= chord; y_u *= chord; x_l *=chord; y_l *=chord

    upper = np.column_stack([x_u, y_u])[::-1] # Reverse the point order TE -> LE
    lower = np.column_stack([x_l, y_l])[1:]
    boundary = np.vstack([upper, lower])

    return boundary, y_t

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test cases
    test_cases = ["1052"]

    for code in test_cases:
        boundary, y_t = gen_naca4(
            code=code,
            chord=1.0,
            n_chord=100
        )

        x, y = boundary[:, 0], boundary[:, 1]
        x_t = np.linspace(0,1, 100)

        plt.figure(figsize=(6, 2))
        plt.plot(x, y, "-k", lw=1)
        plt.plot(x_t, y_t, "-b", lw =1)
        plt.scatter(x[0], y[0], color="red", label="Start point")
        plt.scatter(x[len(x)//2], y[len(x)//2], color="blue", label="Mid point")
        plt.axis("equal")
        plt.title(f"NACA {code}")
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"NACA {code}")
        print(f"  Boundary shape: {boundary.shape}")
        print(f"  First point: {boundary[0]}")
        print(f"  Last point:  {boundary[-1]}")
        print("-" * 40)
   
      


