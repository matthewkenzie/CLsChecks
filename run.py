import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('mphil.mplstyle')
from utils import generate, generate_asimov, fit, plot_fit, test_statistics, plot_ts, plot_spec_ts
from tqdm import tqdm
import itertools

# I'd like to have an argument please
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-p', '--prelim', default=False, action="store_true", help='Run preliminary fits on some toys to see if the initial guess of numbers looks reasonable')
parser.add_argument('-a', '--asimovs', default=False, action="store_true", help='Run fits to asimov toys to compute and store sigma values')
parser.add_argument('-g', '--generate', default=False, action="store_true", help='Re-run the generation of the toys and fitting them back to compute test-statistics')
parser.add_argument('-n','--ntoys', default=10000, type=int, help='Number of toys to run at each point')
parser.add_argument('-b','--Nb', default=5000, type=int, help='Number of background events')
parser.add_argument('-s','--Ns-gen', dest='NsGen', type=int, default=[], nargs="+", action="extend", help='Number of signal events to generate (can be passed multiple times)')
parser.add_argument('-S','--Ns-hyp', dest='NsHyp', type=int, default=[], nargs="+", action="extend", help='Number of signal events to test hypothesis for (can be passed multiple times)')
args = parser.parse_args()

if len(args.NsGen)==0:
    args.NsGen = [0, 10, 20, 25, 50, 75]
if len(args.NsHyp)==0:
    args.NsHyp = [0, 10, 20, 25, 50, 75]

print('NTOYS:', args.ntoys)
print('NBKG:', args.Nb)
print('NSIG (gen):', args.NsGen)
print('NSIG (hyp):', args.NsHyp)

# first want to generate a sample and fit them 
# just to see what looks reasonable
if args.prelim:
    for Ns in args.NsGen:
        toy = generate( Nb=args.Nb, Ns=Ns, poiss=True )
        d2ll, bvals, sbvals, berrs, sberrs = fit( toy, bfit=True, sbfit=True, verbose=2 )
        plot_fit( toy, bvals, sbvals, d2ll, save=f'plots/fit_s{Ns}.pdf' )
    plt.show()

# also want to generate the asimov samples and fit these
if args.asimovs:
    asf = pd.DataFrame( columns=['mu_gen', 'mu_hat', 'sigma'] )
    for i, Ns in enumerate(args.NsGen):
        asimov = generate_asimov( Nb=args.Nb, Ns=Ns )
        d2ll, bvals, sbvals, berrs, sberrs = fit( asimov, bfit=True, sbfit=True, binned=True, verbose=2 )
        plot_fit( asimov, bvals, sbvals, d2ll, binned=True, rebin=8, save=f'plots/asimov_fit_s{Ns}.pdf' )
        asf.loc[i] = [ Ns, sbvals['Ns'], sberrs['Ns'] ]
    
    asf.to_pickle('asimovs.pkl')
    plt.show()
else:
    asf = pd.read_pickle('asimovs.pkl')

# next up we'll run some toys and stash the test-statistic results in a dataframe
if args.generate:
    df = pd.DataFrame( columns=['mu_gen', 'mu_hyp', 'mu_hat', 'n2ll_free', 'n2ll_fixed', 'n2ll_zero', 't_mu', 't_tilde_mu', 'q_zero', 'q_mu', 'q_tilde_mu'] )
    for i in tqdm(range(args.ntoys)):
        for j, mu_gen in enumerate(args.NsGen):
            for k, mu_hyp in enumerate(args.NsHyp):
                toy = generate( Nb=args.Nb, Ns=mu_gen, poiss=True )
                res = test_statistics( toy, mu_hyp, verbose=0 )
                ind = len(df)
                df.loc[ind] = [ mu_gen, mu_hyp, res.mu_hat, res.fmin_free, res.fmin_fixed, res.fmin_zero, res.t_mu, res.t_tilde_mu, res.q_zero, res.q_mu, res.q_tilde_mu ]
    
    df.to_pickle('toys.pkl')

else:
    df = pd.read_pickle('toys.pkl')

# then we can have a look at these different test-statistics
# do a subset at 0, 20, 50
plot_spec_ts( df, 't_mu', [(0,0), (0,50), (50,20), (20, 75)], log=True, pull=False, save='plots/t_mu.png' )
plot_spec_ts( df, 't_tilde_mu', [(0,0), (0,50), (50,20), (20, 75)], log=True, pull=False, save='plots/t_tilde_mu.png' )
plot_spec_ts( df, 'q_zero', [(0,0), (0,50), (50,20), (20, 75)], log=True, pull=False, save='plots/q_zero.png' )
plot_spec_ts( df, 'q_mu', [(0,0), (0,50), (50,20), (20, 75)], log=True, pull=False, save='plots/q_mu.png' )
plot_spec_ts( df, 'q_tilde_mu', [(0,0), (0,50), (50,20), (20, 75)], log=True, pull=False, save='plots/q_tilde_mu.png' )
# for j, mu_gen in enumerate(args.NsGen):
#     for k, mu_hyp in enumerate(args.NsHyp):
#         plot_spec_ts(df, 't_mu',  
        # plot_ts( df, mu_gen, mu_hyp, log=True, interactive=False, save=f'plots/ts_g{mu_gen}_h{mu_hyp}.pdf' )


# t_bonly = []
# t_sb = [ [] for Ns in Nss ]
#
# for i in tqdm(range(ntoys)):
#     btoy = generate(Nb=Nb, Ns=0, poiss=True)
#     d2ll, bvals, sbvals = fit( btoy, bfit=True, sbfit=True, verbose=0 )
#     # plot_fit( toy, bvals, sbvals, d2ll )
#     t_bonly.append( d2ll )
#
#     for j, Ns in enumerate(Nss):
#         sbtoy = generate(Nb=Nb, Ns=Ns, poiss=True)
#         d2ll, bvals, sbvals = fit( sbtoy, bfit=True, sbfit=True, verbose=0 )
#         t_sb[j].append( d2ll )
#
# fig, ax = plt.subplots()
# ax.hist( t_bonly, bins=100, range=(0,100), label='$t(\mu=0)$')
# for Ns, t_vals in zip( Nss, t_sb ):
#     ax.hist( t_vals, bins=100, range=(0,100), label=f'$t(\mu={Ns})$')
# ax.set_xlabel('$t(\mu)$')
# ax.legend()
# plt.show()
