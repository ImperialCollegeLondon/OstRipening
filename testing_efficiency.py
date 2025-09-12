'''import cProfile
import pstats
profiler = cProfile.Profile()
profiler.enable()
self.__elementList__()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()
from IPython import embed; embed()'''
'''
%load_ext line_profiler
'''
'''
from line_profiler import LineProfiler
lp = LineProfiler()
lp_wrapper = lp(compute.isConnected)
lp_wrapper(arrr.copy())
lp.print_stats()
'''


import numpy as np

def brentq_custom(f, a, b, xtol=1e-6, rtol=1e-8, maxiter=100):
    """
    Find a root of f(x) in [a, b] using Brent's method.
    Args:
        f: Function to find root of.
        a, b: Interval where f(a) * f(b) < 0.
        xtol: Absolute tolerance for root.
        rtol: Relative tolerance for root.
        maxiter: Maximum iterations.
    Returns:
        Root estimate, or raises ValueError if convergence fails.
    """
    # Evaluate function at endpoints
    fa = f(a)
    fb = f(b)
    
    # Check if root is bracketed
    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    
    # Initialize variables
    c = a  # Previous point
    fc = fa
    d = b - a  # Previous step
    e = b - a  # Previous-previous step
    
    # Ensure |f(b)| <= |f(a)| to start with b as better estimate
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
        c = a
        fc = fa
    
    iter_count = 0
    while iter_count < maxiter:
        # Check convergence
        tol = xtol + rtol * abs(b)
        if abs(b - a) <= tol or abs(fb) < 1e-15:
            return b
        
        # Decide whether to use bisection
        m = 0.5 * (a - b)  # Half interval width
        bisect = False
        if abs(e) < tol or abs(fa) <= abs(fb):
            bisect = True
        else:
            # Attempt interpolation
            s = fb / fa
            if a == c:  # Secant method
                p = 2.0 * m * s
                q = 1.0 - s
            else:  # Inverse quadratic interpolation
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            
            # Check if interpolation is acceptable
            if p > 0:
                q = -q
            else:
                p = -p
            if (2.0 * p < 3.0 * m * q - abs(tol * q)) and (p < abs(0.5 * e * q)):
                e = d
                d = p / q
            else:
                bisect = True
        
        # Use bisection if interpolation is not used
        if bisect:
            d = m
            e = m
        
        # Update points
        a = b
        fa = fb
        b = b + d
        fb = f(b)
        
        # Ensure root remains bracketed
        if fb * fa > 0:
            c = a
            fc = fa
            d = b - a
            e = d
            a = c
            fa = fc
        
        # Swap to keep |f(b)| <= |f(a)|
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            c = a
            fc = fa
        
        iter_count += 1
    
    raise ValueError(f"Failed to converge after {maxiter} iterations")

# Example usage
def f(x):
    return x**2 - 4  # Root at x = 2

try:
    root = brentq_custom(f, 0, 5, xtol=1e-6, rtol=1e-8)
    print(f"Root: {root:.6f}")
    print(f"Function value at root: {f(root):.6e}")
except ValueError as e:
    print(f"Error: {e}")


