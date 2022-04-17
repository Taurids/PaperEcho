import numpy as np


def norm_pdf(x, loc=0, scale=1):
    """
    Probability mass function at x of the normal distribution function.
    """
    norm_x = (x - loc) / scale
    p = 1 / (np.sqrt(2 * np.pi) * scale)
    p = p * np.exp(-np.square(norm_x) / 2)
    return p / np.sum(p)


def norm2_pdf(x, y, loc_x=0, scale_x=1, loc_y=0, scale_y=1, rho=0.75):
    """
    Probability mass function at x of the bivariate normal distribution function.
    """
    norm_x = (x - loc_x) / scale_x
    norm_y = (y - loc_y) / scale_y
    p = 1 / ((2 * np.pi) * np.sqrt(1 - rho**2))
    plists = []
    for x in norm_x:
        plist = []
        for y in norm_y:
            p = p * np.exp(-(x**2 - 2*rho*x*y + y**2) / (2 - 2*rho**2))
            plist.append(p)
        plists.append(plist)
    return plists / np.sum(plists)
