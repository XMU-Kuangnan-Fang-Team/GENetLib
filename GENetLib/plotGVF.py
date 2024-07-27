from plotFD import plotFD


def plotGVF(x, y = None, xlab = None, ylab = None):
    gvf = x
    if y == None:
        plotFD(gvf, xlab, ylab)
    else:
        plotFD(gvf, y, xlab, ylab)


'''test
from SNPgvf import SNPgvf
from SimDataSNP import SimDataSNP
simdata = SimDataSNP(20,10,'Continuous')
location = list(simdata['location'])
X = simdata['X']
SNPgvf_res = SNPgvf(location, X, btype = "Bspline", nbasis = 5, params = 4, Plot = False)
plotGVF(SNPgvf_res)'''