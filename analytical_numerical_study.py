import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc


class plotClass:
    def __init__(self, network, file, shape, D=7.3e-9):
        self.network = network
        self.file = file
        self.x_values = np.loadtxt('coordinates_{}.dat'.format(network))[:,0]
        self.conc = np.memmap('gasConcOstRipening_{}.dat'.format(file), dtype='float32', 
                              mode='r', shape=shape)
        #from IPython import embed; embed()
        self.pc = np.memmap('clustPcOstRipening_{}.dat'.format(file), dtype='float32', 
                              mode='r', shape=(shape[0],))
        self.t_values = np.loadtxt('timeArray_{}.dat'.format(network))
        self.D = D
        self.C0 = self.conc[0]
        self.colorlist = colorlist=['b','k','r','y','c','olive','violet','m','g','tomato']

    def plot_conc_t(self):
        # Plotting the concentration profile vs position at different times
        plt.figure(figsize=(8, 6))

        leg = []
        nbins = 10
        xbin = np.linspace(0, self.x_values.max(), 10)
        x_values = []
        for i in range(1,nbins):
            x_values.append(np.where((self.x_values>xbin[i-1])&(self.x_values<xbin[i]))[0][0])

        for ind, x in enumerate(x_values):
            C_numerical = self.conc[:, x]
            plt.plot(self.t_values, C_numerical, 'o', color=self.colorlist[ind])
            leg.append(f'x = {round(x,3)} m')

        xlabel='Time t(s)'
        ylabel='Concentration (mol/m3)'
        xlim = [0.0, None]
        ylim = [None, None]
        formatFig(xlabel, ylabel, leg, xlim, ylim)
        plt.savefig(f'diffusion_conc_time_{self.network}.png', dpi=500)
        plt.close()

    def plot_conc_x(self):
        # Plotting the concentration profile vs time at different positions
        plt.figure(figsize=(8, 6))

        leg = []
        nbins = 10
        tbin = np.linspace(0, self.t_values.max(), 10)
        t_values = []
        for i in range(1,nbins):
            t_values.append(np.where((self.t_values>tbin[i-1])&(self.t_values<tbin[i]))[0][0])

        for ind, t in enumerate(t_values):
            C_numerical = self.conc[t,1:self.x_values.size-1]
            plt.plot(self.x_values[1:-1], C_numerical, 'o', color=self.colorlist[ind])
            leg.append(f'x = {round(t,3)} s')

        xlabel='Position x(m)'
        ylabel='Concentration (mol/m3)'
        xlim = [0.0, None]
        ylim = [None, None]
        formatFig(xlabel, ylabel, leg, xlim, ylim)
        plt.savefig(f'diffusion_conc_position_{self.network}.png', dpi=500)
        plt.close()

    def plot_conc_universal(self):
        plt.figure(figsize=(8, 6))

        leg = []
        nbins = 10
        tbin = np.linspace(0, self.t_values.max(), 10)
        t_values = []
        for i in range(1,nbins):
            t_values.append(np.where((self.t_values>tbin[i-1])&(self.t_values<tbin[i]))[0][0])

        for ind, x in enumerate(self.x_values):
            C_numerical = self.conc[t_values, x]
            plt.plot(t_values, C_numerical, 'o', color=self.colorlist[ind])
            leg.append(f'x = {round(x,3)} m')

        xlabel='Position x(m)'
        ylabel='Concentration (mol/m3)'
        xlim = [0.0, None]
        ylim = [None, None]
        formatFig(xlabel, ylabel, leg, xlim, ylim)
        plt.savefig(f'diffusion_conc_position_{self.network}.png', dpi=500)
        plt.close()

    def plot_avgConc_t(self):
        # Plotting the average concentration profile vs position at different times
        plt.figure(figsize=(8, 6))

        leg = []
        nbins = 10
        #x_values = self.values[1:]
        xbin = np.linspace(0, self.x_values[1:-1].max(), nbins)
        #print(xbin)
        #from IPython import embed; embed()
        
        for ind, i in enumerate(range(1,nbins)):
            xarr = np.where(((self.x_values[1:-1]>xbin[i-1])&(self.x_values[1:-1]<xbin[i])))[0]
            x = self.x_values[xarr+1].mean()
            
            C_numerical = self.conc[:, xarr+1].mean(axis=1)
            plt.plot(self.t_values[:43186], C_numerical[:43186], color=self.colorlist[ind])
            leg.append(f'x = {round(x,6)} m')

        xlabel='Time t(s)'
        ylabel='Concentration (mol/m3)'
        xlim = [0.0, None]
        ylim = [None, None]
        plt.xscale('log')
        formatFig(xlabel, ylabel, leg, xlim, ylim)
        plt.savefig(f'diffusion_avgconc_time_{self.file}.png', dpi=500)
        plt.close()



       

def diffusion_solution(x, t, D, C0):
    return C0 * erfc(x / (2 * np.sqrt(D * t)))

def diffusion_solution_universal(eta, C0):
    return C0 * erfc(eta)


def formatFig(xlabel, ylabel, leg, xlim, ylim, loc=1, bbox_to_anchor=None):
    label_font = {'fontname': 'Arial', 'size': 14, 'color': 'black', 'weight': 'bold',
                    'labelpad': 10}
    
    legend = plt.legend(leg, frameon=False,loc=loc, bbox_to_anchor=bbox_to_anchor)
    ax = plt.gca()
    ax.tick_params(direction='in', axis='both', which='major', pad=10)
    
    # Set the rotation and label coordinates for the fraction part
    #ax.yaxis.label.set_rotation(0)
    #ax.yaxis.set_label_coords(-0.19, 0.5)

    # Increase font size and make it bold        
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() 
                    + ax.get_yticklabels() + legend.get_texts()):
        item.set_fontsize(14)
        item.set_fontname('Arial')
        item.set_fontweight('bold')
        
    # Remove right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel(xlabel, **label_font)
    ax.set_ylabel(ylabel, **label_font, rotation=90)
    
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    plt.tight_layout()
        
#pC = plotClass('bent', 'bent_0', (43201,27329))
#pC.plot_avgConc_t()