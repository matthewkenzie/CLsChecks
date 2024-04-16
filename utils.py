import numpy as np
import pandas as pd
from iminuit import cost, Minuit
from models import sgen, bgen, true_theta, xrange, tpdf, tdensity, tintegral, test_stat_asymp_dists 
import matplotlib.pyplot as plt

test_stat_titles = {
    't_mu': r'$t_\mu$',
    't_tilde_mu': r'$\tilde{t}_\mu$',
    'q_zero': r'$q_0$',
    'q_mu': r'$q_\mu$',
    'q_tilde_mu': r'$\tilde{q}_\mu$'
}

def generate(Nb, Ns, poiss=True):

    if poiss:
        Ns = np.random.poisson(Ns)
        Nb = np.random.poisson(Nb)

    sevs = sgen( Ns, 
                 mu = true_theta['mu'], 
                 sg = true_theta['sg'], 
                 a = xrange[0], 
                 b = xrange[1] ) 

    bevs = bgen( Nb,
                 lb = true_theta['lb'],
                 a = xrange[0],
                 b = xrange[1] )

    return np.concatenate( [sevs, bevs] )

def generate_asimov(Nb, Ns, bins=400):

    xe = np.linspace(*xrange, bins+1)
    cx = 0.5*(xe[1:]+xe[:-1])
    bw = xe[1]-xe[0]
    y = bw * tpdf(cx, Ns, Nb, true_theta['mu'], true_theta['sg'], true_theta['lb'])
    return xe, y


def plot_fit( data, bfit_vals={}, sbfit_vals={}, d2ll_val=None, binned=False, rebin=0, save=None ):

    fig, ax = plt.subplots(2, 1, figsize=(6.4,6.4), gridspec_kw=dict(hspace=0, height_ratios=(3,1)))
    
    if binned:
        xe = data[0]
        nh = data[1]
        if rebin>0:
            xe = np.concatenate( [ xe[:-1].reshape((-1,rebin))[:,0], [xe[-1]] ] ) 
            nh = np.sum(nh.reshape((-1,rebin)),axis=1)
    else:
        nh, xe = np.histogram(data, bins=50)
    
    cx = 0.5 * ( xe[:-1] + xe[1:] )
    bw = xe[1]-xe[0]

    ax[0].errorbar( cx, nh, nh**0.5, fmt='ko', label='Data' )

    # for pull
    x = np.linspace(*xrange, 400)
    ax[1].plot(x, np.zeros_like(x), 'k-')

    if len(bfit_vals)>0:
        x = np.linspace(*xrange, 400, endpoint=False)
        y = bw*tpdf(x, **bfit_vals)
        ax[0].plot(x, y, 'r--', label='B only Fit')
        # for pull
        cy = bw*tpdf(cx, **bfit_vals)
        py = (nh-cy)/(nh**0.5)
        ax[1].errorbar( cx, py, np.ones_like(cx), fmt='ro' )

    if len(sbfit_vals)>0:
        x = np.linspace(*xrange, 400, endpoint=False)
        y = bw*tpdf(x, **sbfit_vals)
        ax[0].plot(x, y, 'b-', label='S+B Fit')
        # for pull
        cy = bw*tpdf(cx, **sbfit_vals)
        py = (nh-cy)/(nh**0.5)
        ax[1].errorbar( cx, py, np.ones_like(cx), fmt='bo', markerfacecolor="none" )

    if d2ll_val is not None:
        ax[0].text(0.01, 0.01, rf'$t = -2\ln\left( \frac{{ L(\mu=0)}}{{L(\mu=\hat{{\mu}})}} \right) = {d2ll_val:5.3f}$', va='bottom', ha='left', transform=ax[0].transAxes)
    
    ax[0].set_xlim(*xrange)
    ax[0].set_ylim(bottom=0)
    ax[0].set_xticklabels([])
    ax[1].set_xlim(*xrange)
    ylim = ax[1].get_ylim()
    maxy = np.max( np.abs(ylim) )
    ax[1].set_ylim( -maxy, maxy )
    ax[0].legend()
    ax[1].set_xlabel('Fit Variable')
    ax[1].set_ylabel('Pull')
    ax[0].set_ylabel('Events')

    if save is not None:
        fig.savefig(save)

def init_fit(data, binned=False):
    
    if binned:
        xe = data[0]
        w = data[1]
        w2 = data[1]
        nh = np.stack((w,w2),axis=1)
        n2ll = cost.ExtendedBinnedNLL( nh, xe, tintegral )
        nevs = sum(w)

    else:
        n2ll = cost.ExtendedUnbinnedNLL( data, tdensity )
        nevs = len(data)
    
    start_vals = { 'Ns': 0,
                   'Nb': nevs,
                   'mu': true_theta['mu'],
                   'sg': true_theta['sg'],
                   'lb': true_theta['lb'] }

    mi = Minuit( n2ll, **start_vals )

    # set parameter limits 
    mi.limits['Ns'] = (-nevs, nevs)
    mi.limits['Nb'] = (0, 1.5*nevs)
    mi.limits['mu'] = xrange
    mi.limits['sg'] = (0.01*true_theta['sg'], 4*true_theta['sg'])
    mi.limits['lb'] = (0.01*true_theta['lb'], 50*true_theta['lb'])
    
    # fix peak position and width
    mi.fixed['mu'] = True
    mi.fixed['sg'] = True

    return n2ll, mi

## note mu here is the mu of the Cowan paper
## which in our case is Ns
def test_statistics(data, mu, verbose=0):
    
    # make an object to return the result
    class TestStatisticsResult(object):
        pass
    
    res = TestStatisticsResult()


    n2ll, mi = init_fit( data )

    # run fit with mu free 
    mi.fixed['lb'] = False
    mi.fixed['Nb'] = False
    mi.fixed['Ns'] = False

    mi.migrad()
    mi.hesse()
    res.fmin_free = mi.fval
    res.mu_hat = mi.values['Ns']

    if verbose>0:
        print('Fit with mu free')
        if verbose==1:
            print(mi.params)
        else:
            print(mi)

    # run fit with mu fixed to mu val
    mi.values['Ns'] = mu
    mi.fixed['Ns'] = True

    mi.migrad()
    mi.hesse()
    res.fmin_fixed = mi.fval

    if verbose>0:
        print(f'Fit with mu={mu}')
        if verbose==1:
            print(mi.params)
        else:
            print(mi)

    # run fit with mu fixed to zero
    mi.values['Ns'] = 0
    mi.fixed['Ns'] = True

    mi.migrad()
    mi.hesse()
    res.fmin_zero = mi.fval

    if verbose>0:
        print('Fit with mu=0')
        if verbose==1:
            print(mi.params)
        else:
            print(mi)

    # test statistics

    # Eq. 8
    res.t_mu = res.fmin_fixed - res.fmin_free

    # Eq. 11
    res.t_alt = res.fmin_fixed - res.fmin_zero
    res.t_tilde_mu = res.t_mu if res.mu_hat >= 0 else res.t_alt

    # Eq. 12
    res.q_zero = res.fmin_zero - res.fmin_free if res.mu_hat >= 0 else 0

    # Eq. 14
    res.q_mu = res.t_mu if res.mu_hat <= mu else 0

    # Eq. 16
    res.q_tilde_mu = res.t_tilde_mu if res.mu_hat <= mu else 0
    
    return res
    

def fit(data, bfit=True, sbfit=True, binned=False, verbose=0):

    n2ll, mi = init_fit( data, binned )

    if sbfit:

        mi.fixed['lb'] = False
        mi.fixed['Ns'] = False
        mi.fixed['Nb'] = False

        mi.migrad()
        mi.hesse()

        sbfit_n2ll = mi.fval
        sbfit_vals = mi.values.to_dict()
        sbfit_errs = mi.errors.to_dict()

        if verbose>0:
            print('SB Fit:')
            if verbose==1:
                print(mi.params)
            else:
                print(mi)

    if bfit:
        mi.values['Ns'] = 0
        mi.fixed['Ns'] = True
        mi.fixed['Nb'] = False
        mi.fixed['lb'] = False

        mi.migrad()
        mi.hesse()

        bfit_n2ll = mi.fval
        bfit_vals = mi.values.to_dict()
        bfit_errs = mi.errors.to_dict()

        if verbose>0:
            print('SB Fit:')
            if verbose==1:
                print(mi.params)
            else:
                print(mi)

    if bfit and sbfit:

        d2ll = bfit_n2ll - sbfit_n2ll

        return d2ll, bfit_vals, sbfit_vals, bfit_errs, sbfit_errs

    elif bfit and not sbfit:
        return bfit_vals, bfit_errs

    elif sbfit and not bfit:
        return sbfit_vals, sbfit_errs

    else:
        return None

def plot_ts(df, mu_gen, mu_hyp, log=False, interactive=True, save=None):
    fig, ax = plt.subplots()
    
    pf = df.query( f'mu_gen=={mu_gen} & mu_hyp=={mu_hyp}' )
    for test_stat, title in test_stat_titles.items():
        ax.hist( pf[test_stat], bins=100, histtype='step', density=True, label=title )
    ax.text( 0.98, 0.50, f'$\mu_{{gen}} = {mu_gen:2d}$', ha='right', va='center', transform=ax.transAxes )
    ax.text( 0.98, 0.44, f'$\mu_{{hyp}} = {mu_hyp:2d}$', ha='right', va='center', transform=ax.transAxes )
    
    ax.legend()
    ax.set_xlabel('Test Statistic')
    if log:
        ax.set_yscale('log')
    if save is not None:
        fig.savefig(save)
    if interactive:
        plt.show()
    else:
        plt.close()

def plot_spec_ts(df, test_statistic, mu_mup_pairs=[], draw_asymptotic=True, log=False, pull=False, interactive=True, save=None):
    """
    note that mu_mup_pairs expect a list of tuples with 2 values - the mu, mu_prime (i.e. mu=mu_hyp, mu_prime=mu_gen)
    the sigma will be looked up from the asimovs fit file
    """
    
    assert( test_statistic in test_stat_titles.keys() )
    if draw_asymptotic and pull:
        fig, axes = plt.subplots(2, 1, figsize=(6.4,6.4), gridspec_kw=dict(hspace=0, height_ratios=(3,1)))
        ax = axes[0]
        pax = axes[1]
        pax.set_xlabel( test_stat_titles[test_statistic] )
    else:
        fig, ax = plt.subplots()
        ax.set_xlabel( test_stat_titles[test_statistic] )

    for i, (mu, mu_prime) in enumerate(mu_mup_pairs):
        pf = df.query( f'mu_gen=={mu_prime} & mu_hyp=={mu}' )
        # draw it
        nh, xe, _ = ax.hist( pf[test_statistic], bins=100, histtype='step', density=True, label=f'$\mu\'={mu_prime}, \mu={mu}$')

        # histogram it to keep weights if we want to draw as point
        w, _ = np.histogram( pf[test_statistic], bins=xe )
        err = (w**0.5/w)*nh
        bw = xe[1] - xe[0]
        cx = 0.5*(xe[:-1]+xe[1:])
        
        if draw_asymptotic:
            # look up asimov sigmas
            asf = pd.read_pickle('asimovs.pkl')
            sigma = asf.query( f'mu_gen=={mu}')['sigma'].to_numpy()[0]

            x = np.linspace(xe[0],xe[-1], 400)
            y = test_stat_asymp_dists[test_statistic]( x, mu, mu_prime, sigma )
            ax.plot( x, y, c=f'C{i}' )

            # pull
            if pull:
                pax.plot( x, np.zeros_like(x), 'k-' )
                cy = test_stat_asymp_dists[test_statistic]( cx, mu, mu_prime, sigma )
                py = ( nh - cy ) / err
                pax.errorbar( cx, py, np.ones_like(py), marker='.', color=f'C{i}', ls='none' )
                pax.set_ylim(-5,5)


    ax.legend()
    
    if log:
        ax.set_yscale('log')
    if save is not None:
        fig.savefig(save)
    if interactive:
        plt.show()
    else:
        plt.close()
