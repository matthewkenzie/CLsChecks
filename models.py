import numpy as np
from numba_stats import truncnorm, truncexpon, norm

# define the range
xrange = (5, 5.6)

# set true parameters
true_theta = { 'f': 0.1,
               'mu': 5.28,
               'sg': 0.018,
               'lb': 2,
             }

## fast signal pdf from numba-stats ##
def spdf(x, mu, sg, a=xrange[0], b=xrange[1]):
    return truncnorm.pdf(x, a, b, mu, sg)

## fast signal cdf from numba-stats ##
def scdf(xe, mu, sg, a=xrange[0], b=xrange[1]):
    return truncnorm.cdf(xe, a, b, mu, sg)

## fast background pdf from numba-stats ##
def bpdf(x, lb, a=xrange[0], b=xrange[1]):
    return truncexpon.pdf(x, a, b, a, 1/lb)

## fast background cdf from numba-stats ##
def bcdf(xe, lb, a=xrange[0], b=xrange[1]):
    return truncexpon.cdf(xe, a, b, a, 1/lb)

## fast total pdf ##
def tpdf(x, Ns, Nb, mu, sg, lb, comps=['s','b']):
    ret = np.zeros_like(x)
    if 's' in comps:
        ret += Ns * spdf(x, mu, sg)
    if 'b' in comps:
        ret += Nb * bpdf(x, lb)
    return ret

## fast total cdf ##
def tcdf(xe, Ns, Nb, mu, sg, lb, comps=['s','b']):
    ret = np.zeros_like(xe)
    if 's' in comps:
        ret += Ns * scdf(xe, mu, sg)
    if 'b' in comps:
        ret += Nb * bcdf(xe, lb)
    return ret

## fast total density ##
def tdensity(x, Ns, Nb, mu, sg, lb):
    return Ns + Nb, tpdf(x, Ns, Nb, mu, sg, lb)

## fast total integral ##
def tintegral(xe, Ns, Nb, mu, sg, lb):
    return tcdf(xe, Ns, Nb, mu, sg, lb)

## fast signal generate from numba-stats ##
def sgen(size, mu, sg, a=xrange[0], b=xrange[1]):
    return truncnorm.rvs(a, b, mu, sg, size, random_state=None)

## fast background generate from numba-stats ##
def bgen(size, lb, a=xrange[0], b=xrange[1]):
    return truncexpon.rvs(a, b, a, 1/lb, size, random_state=None)

## fast total generate ##
def tgen(size, f, mu, sg, lb, a=xrange[0], b=xrange[1], poiss=True):
    Ns = f*size
    Nb = (1-f)*size
    if poiss:
        Ns = np.random.poisson(Ns)
        Nb = np.random.poisson(Nb)
    
    sevs = sgen(Ns, mu, sg, a, b)
    bevs = bgen(Nb, lb, a, b)
    return np.concatenate( [sevs,bevs] )

## asymptotic test-statistic distributions from arXiv:1007.1727
def t_mu_dist(t_mu, mu, mu_prime=None, sigma=None):
    """
    Note that mu here is the hypothesised value of mu.
    mu_prime is the true or generated value of mu.
    sigma is the uncertainty on my when fitting the Asimov
    """

    # when mu=mu'
    if mu_prime is None and sigma is None:
        return (1/np.sqrt(2*np.pi*t_mu)) * np.exp( -t_mu / 2 )
    elif mu_prime is not None and sigma is not None:
        neu = (mu - mu_prime)/sigma
        tpneu = t_mu**0.5 + neu
        tmneu = t_mu**0.5 - neu
        return ( np.exp( -tpneu**2 / 2 ) + np.exp( -tmneu**2 / 2 ) ) / ( 2*np.sqrt(2*np.pi*t_mu) ) 

def t_tilde_mu_dist(t_tilde_mu, mu, mu_prime=None, sigma=None):
    """
    Note that mu here is the hypothesised value of mu.
    mu_prime is the true or generated value of mu.
    sigma is the uncertainty on my when fitting the Asimov
    """

    muosg2 = mu**2 / sigma**2
    # when mu=mu'
    if (mu_prime is None and sigma is None) or mu==mu_prime:

        denom = np.nan_to_num( np.sqrt( 2*np.pi* t_tilde_mu ) )
        term1 = np.exp( -0.5*t_tilde_mu ) / denom 
        
        # term 2 only for t_tilde_mu > muosg2
        ttgt = t_tilde_mu[ t_tilde_mu > muosg2 ]
        
        # term2 = 0 if mu==0
        if mu==0:
            term2 = np.zeros_like( ttgt )
        else:
            term2 = np.exp( -0.5*(ttgt + muosg2)**2 / (4*muosg2) ) / ( np.sqrt( 2*np.pi ) * (2*mu/sigma) )


        ret = np.zeros_like( t_tilde_mu )
        ret[ t_tilde_mu <= muosg2 ] = term1[ t_tilde_mu <= muosg2 ]
        ret[ t_tilde_mu > muosg2 ] = 0.5*term1[ t_tilde_mu > muosg2 ] + term2
        
        term2 = np.zeros_like( t_tilde_mu[ t_tilde_mu > muosg2 ] )
        
        return ret
    
    elif mu_prime is not None and sigma is not None:
        
        neu = (mu - mu_prime) / sigma
        tpneu = t_tilde_mu**0.5 + neu
        tmneu = t_tilde_mu**0.5 - neu
        neu_tilde = (mu**2 - 2*mu*mu_prime) / sigma**2
        
        # first term is always there
        term1 = np.exp( -tpneu**2 / 2 ) / ( 2*np.sqrt(2*np.pi*t_tilde_mu) )
        
        # second term depends on t_tilde_mu value
        term2 = np.zeros_like( t_tilde_mu )
        muosg2 = mu**2 / sigma**2
        ttlt = t_tilde_mu[ t_tilde_mu <= muosg2 ]
        ttgt = t_tilde_mu[ t_tilde_mu > muosg2 ]
        
        # less than piece
        tmneu = ttlt**0.5 - neu
        term2[ t_tilde_mu <= muosg2 ] = np.exp( -tmneu**2 / 2 ) / ( 2*np.sqrt(2*np.pi*ttlt) )

        # greater than piece
        if mu==0:
            term2[ t_tilde_mu > muosg2 ] = np.zeros_like( ttgt )
        else:
            term2[ t_tilde_mu > muosg2 ] = np.exp( -0.5*( ttgt - neu_tilde )**2 / ( 4*muosg2 ) ) /  ( np.sqrt( 2*np.pi ) * (2*mu/sigma) )

        return term1 + term2 

def q_zero_dist(q_zero, mu, mu_prime=None, sigma=None):
    """
    note that it does not depend on mu
    """
    if mu_prime==0 or mu_prime is None or sigma is None:
        ret = np.zeros_like( q_zero )
        ret[ q_zero==0 ] += 0.5 
        ret += 0.5 * np.exp( -q_zero/2 ) / np.sqrt( 2*np.pi*q_zero )
        return ret
    else:
        mupos = mu_prime / sigma
        ret = np.zeros_like( q_zero )
        ret[ q_zero==0 ] += 1 - norm.cdf( mupos, 0, 1 ) 
        ret += 0.5 * np.exp( -0.5*( q_zero**0.5 - mupos )**2 ) / ( np.sqrt( 2*np.pi*q_zero ) )
        return ret

def q_mu_dist(q_mu, mu, mu_prime=None, sigma=None):
    """
    Note that mu here is the hypothesised value of mu.
    mu_prime is the true or generated value of mu.
    sigma is the uncertainty on my when fitting the Asimov
    """

    # when mu==mu'
    if (mu_prime is None and sigma is None) or mu==mu_prime:
        ret = np.zeros_like( q_mu )
        ret[ q_mu==0 ] += 0.5 
        ret += 0.5 * np.exp( -q_mu/2 ) / np.sqrt( 2*np.pi*q_mu )
        return ret

    else:
        ret = np.zeros_like( q_mu )
        neu = ( mu_prime - mu ) / sigma
        ret[ q_mu==0 ] += norm.cdf( neu, 0, 1 )
        ret += 0.5 * np.exp( -0.5*( q_mu**0.5 + neu )**2 ) / np.sqrt( 2*np.pi*q_mu )
        return ret

def q_tilde_mu_dist(q_tilde_mu, mu, mu_prime=None, sigma=None):
    """
    Note that mu here is the hypothesised value of mu.
    mu_prime is the true or generated value of mu.
    sigma is the uncertainty on my when fitting the Asimov
    """

    # when mu==mu'
    if (mu_prime is None and sigma is None) or mu==mu_prime:
        ret = np.zeros_like( q_tilde_mu )
        ret[ q_tilde_mu==0 ] += 0.5 
        muosg2 = mu**2 / sigma**2
        
        qmult = q_tilde_mu[ q_tilde_mu <= muosg2 ]
        qmugt = q_tilde_mu[ q_tilde_mu > muosg2 ]

        # less than piece
        ret[ q_tilde_mu <= muosg2 ] += 0.5 * np.exp( -qmult / 2 ) / np.sqrt( 2*np.pi*qmult )

        # greater than piece
        if mu==0:
            term2 = np.zeros_like( qmugt )
        else:
            term2 = np.exp( -0.5 * ( qmugt + muosg2 )**2 / ( 4*muosg2 ) ) / ( 2*np.sqrt( 2*np.pi)*mu/sigma ) 

        ret [ q_tilde_mu > muosg2 ] += term2

        return ret

    else:
        ret = np.zeros_like( q_tilde_mu )
        neu = ( mu_prime - mu ) / sigma
        ret[ q_tilde_mu==0 ] += norm.cdf( neu, 0, 1 )
        muosg2 = mu**2 / sigma**2
        neu_tilde = (mu**2 - 2*mu*mu_prime) / sigma**2

        qmult = q_tilde_mu[ q_tilde_mu <= muosg2 ]
        qmugt = q_tilde_mu[ q_tilde_mu > muosg2 ]

        # less than piece
        ret[ q_tilde_mu <= muosg2 ] += 0.5 * np.exp( -0.5*( qmult**0.5 + neu )**2 ) / np.sqrt( 2*np.pi*qmult )

        # greater than piece
        if mu==0:
            term2 = np.zeros_like( qmugt )
        else:
            term2 = np.exp( -0.5 * ( qmugt - neu_tilde )**2 / ( 4*muosg2 ) ) / ( 2*np.sqrt( 2*np.pi)*mu/sigma ) 

        ret [ q_tilde_mu > muosg2 ] += term2

        return ret

test_stat_asymp_dists = {
    't_mu': t_mu_dist,
    't_tilde_mu': t_tilde_mu_dist,
    'q_zero': q_zero_dist,
    'q_mu': q_mu_dist,
    'q_tilde_mu': q_tilde_mu_dist
}


