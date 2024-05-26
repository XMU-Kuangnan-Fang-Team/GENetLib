from plotFD import plotFD


def plotGVF(x, y = None, xlab = None, ylab = None):
    gvf = x
    if y == None:
        plotFD(gvf, xlab, ylab)
    else:
        plotFD(gvf, y, xlab, ylab)

