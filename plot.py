import pandas as pd
import numpy as np
from matplotlib import markers, axes, pyplot as plt
import os
pd.options.mode.chained_assignment = None

class makePlot():
    muw, munw, sigma = 0.821e-3, 0.838e-3, 57e-3
    lnetwork, absK, absK0 = 3.0035e-3, 2.7203e-12, 1.85e-12
    area = lnetwork**2
    lmean, rmean = 0.0001973, 2.2274e-05
    por = 0.2190

    def __init__(self, num, title,  results,
                 compWithLitData=False, compWithPrevData=False, drain=False, imbibe=False, exclude=None, include=None, hysteresis=False, includeTrapping=True, scaled=False):

        self.colorlist = ['g', 'r', 'b', 'gold', 'k', '#b35806', '#542788', 'lime', 'm', 'c',  
                          'lightcoral', 'navy', 'tomato', 'khaki', 'olive', 'gold', 'teal', 'darkcyan', 'tan', 'limegreen']
        #self.colorlist = ['#b2182b', '#ef8a62', '#fddbc7', '#d1e5f0', '#67a9cf', '#2166ac']
        self.markerlist = ['v', '^', '<', '>', 'p', 'P','d', 'D', 'h', 'H', 's', 'o', 'v', '^', 
                           's', 'v', 'o', '^',  'd', 'D', 'h', 'H']
        self.linelist = ['--', '-', '-.', ':', (0, (2, 2)), (0, (1, 2)), (0, (1, 3)), '-', 
                         (0, (5, 10)), (0, (5, 1)), 
                         (0, (3, 1, 1, 1)), (0, (3, 10, 1, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]
        self.num = num
        self.title = title
        
        self.compWithLitData = compWithLitData
        self.compWithPrevData = compWithPrevData
        self.drain = drain
        self.imbibe = imbibe
        self.exclude = exclude
        self.include = include
        self.results = results
        self.hysteresis = hysteresis
        self.includeTrapping = includeTrapping
        if includeTrapping:
            self.label = 'wt'
            self.description = 'percolation with trapping'  
        else:
            self.label = 'nt'
            self.description = 'percolation without trapping'
            self.compWithLitData = False

        self.tuneParam = 0.1
        self.scaled = scaled
        self.img_dir = "./result_images/"
        os.makedirs(os.path.dirname(self.img_dir), exist_ok=True)

        #from IPython import embed; embed()
        
        if self.drain:
            drainageBank(self)
        if self.imbibe:
            imbibitionBank(self)

        

        

    def formatFig(self, xlabel, ylabel, leg, xlim, ylim, loc=1):
        label_font = {'fontname': 'Arial', 'size': 14, 'color': 'black', 'weight': 'bold',
                      'labelpad': 10}
        
        legend = plt.legend(leg, frameon=False,loc=loc)
        print(legend)
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
            #item.set_w

        # Remove right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xlabel(xlabel, **label_font)
        ax.set_ylabel(ylabel, **label_font, rotation=90)
        
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

        plt.tight_layout()

            
    def pcSw(self):
        filename = self.img_dir+'Pc_vs_Sw_hysteresis_{}_{}_{}.png'.format(
            self.title, self.label, self.num)
        
        leg = []
        ind, j = 0, 0
        isFirstCycle = True

        for val1 in self.results.keys():
            for val2 in self.results[val1].keys():
                res = self.results[val1][val2]
                res['satW1']=(res['satW']+self.tuneParam)/(1+self.tuneParam)
                try:
                    assert isFirstCycle
                    plt.plot(res['satW1'], res['capPres']/1000, self.linelist[j],
                            color=self.colorlist[j], linewidth=2)
                    leg.append('cycle_{}'.format(j+1))
                except AssertionError:
                    plt.plot(res['satW1'], res['capPres']/1000, self.linelist[j],
                            color=self.colorlist[j], linewidth=2, label = '_nolegend_')
               
                j += 1
            ind += 1
            j = 0
            isFirstCycle = False
            
        xlabel = r'Sw'
        ylabel = r'Capillary Pressure ($\boldsymbol{kPa}$)'
        xlim = (0, 1.01)
        ylim = (0 , 25)
        self.formatFig(xlabel, ylabel, leg, xlim, ylim)
        plt.savefig(filename, dpi=500)
        plt.close()


    def pcSwScaled(self):
        filename = self.img_dir+'Pc_vs_Se_hysteresis_{}_scaled.png'.format(
            self.title)
        
        leg = []
        ind, indi, j = 0, 0, 0
        isFirstCycle = True
        linelist = {'drainage': {'wt':'-', 'nt':':'}, 'imbibition': {'wt':'-', 'nt':':'}}
        colorlist = {'drainage': {'wt':'k', 'nt':'r'}, 
                     'imbibition': {'wt':'b', 'nt':'brown'}
                    }
        markerlist = {'drainage': {'wt':'s', 'nt':'<'}, 'imbibition': {'wt':'o', 'nt':'v'}}
        
        q=0.4
        for label in ['wt','nt']:
            isFirstCycle = True

            Swi=((self.results['drainage'][label+'_cycle1']['satW']+self.tuneParam)/(
                1+self.tuneParam)).min()
            Sgr = 1-((self.results['imbibition'][label+'_cycle1']['satW']+self.tuneParam)/(
                1+self.tuneParam)).max()
            label1 = 'trapping' if label=='wt' else 'no trapping'
            
            print(label, Swi, Sgr)


            for val1 in ['drainage', 'imbibition']:
                isFirstCycle = True
                for val2 in [label+'_cycle1', label+'_cycle2', label+'_cycle3']:
                    #from IPython import embed; embed()
                    try:
                        assert j!=0 or val1=='imbibition'
                        print(val1, val2)
                        res = self.results[val1][val2]
                        res['satW1']=(res['satW']+self.tuneParam)/(1+self.tuneParam)
                        res['Se'] = (res['satW1']-Swi)/(1-Swi-Sgr)

                        try:
                            assert isFirstCycle
                            plt.plot(res['Se'], res['capPres']/1000, 
                                    linestyle=linelist[val1][label],
                                    marker=markerlist[val1][label], markevery=0.2,
                                    markersize=10,
                                    color=colorlist[val1][label],
                                    #color=colorlist[val1],
                                    linewidth=3)
                            leg.append('secondary {} ({})'.format(val1, label1))
                            isFirstCycle = False
                            
                        except AssertionError:
                            plt.plot(res['Se'], res['capPres']/1000, 
                                    linestyle=linelist[val1][label],
                                    color=colorlist[val1][label],
                                    marker=markerlist[val1][label], markevery=0.2,
                                    markersize=10,
                                    #color=colorlist[val1],
                                    linewidth=3, label='_nolegend_')
                    except AssertionError:
                        pass
                
                    j += 1
                    ind += 1
                j, ind = 0, indi

                
            indi = 3
            ind = indi
            q=0.6
            
        xlabel = r'$\mathbf{S_e}$'
        ylabel = r'Capillary Pressure ($\boldsymbol{kPa}$)'
        xlim = (0, 1.01)
        ylim = (0 , 25)
        #plt.text(x, y, s, fontsize=12)
        self.formatFig(xlabel, ylabel, leg, xlim, ylim)
        plt.savefig(filename, dpi=500)
        plt.close()


    def pcSw1(self):
        filename = self.img_dir+'Pc_vs_Sw_hysteresis_{}_{}_{}_wld.png'.format(
            self.title, self.label, self.num)
        
        leg = []
        ind, j = 0, 0
        isFirstCycle = True
        linelist = {'drainage': '--', 'imbibition': '-.'}
        colorlist = {'drainage': 'r', 'imbibition': 'b'}

        for val1 in self.results.keys():
            if val1 != 'Literature data':
                alreadyDone = False
                for val2 in self.results[val1].keys():
                    res = self.results[val1][val2]
                    print(val1, val2)
                    res['satW1']=(res['satW']+self.tuneParam)/(1+self.tuneParam)
                    
                    try:
                        assert isFirstCycle
                        try:
                            assert self.includeTrapping
                            plt.plot(res['satW1'], res['capPres']/1000, linestyle='-',
                                    color='k', linewidth=2)
                            leg.append('primary drainage')

                            res1=res['satW1'].loc[res['capPres']<25000]
                            i = np.argmin(abs(res['satW1'] - np.median(res1)))
                            x0, y0 = res['satW1'].iloc[i], res['capPres'].iloc[i]/1000
                            x1, y1 = res['satW1'].iloc[i+1], res['capPres'].iloc[i+1]/1000
                            plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                        arrowprops=dict(arrowstyle="-|>, head_length=0.7", facecolor='black', mutation_scale=25), label='_nolegend_')
                            plt.annotate(self.description, xy=(0.3,12.0), label='_nolegend_',
                                        weight='bold', fontsize=14)
                        except AssertionError:
                            pass
                        
                        isFirstCycle = False
                    except AssertionError:
                        try:
                            assert not alreadyDone
                            plt.plot(res['satW1'], res['capPres']/1000, linestyle=linelist[val1],
                                    color=colorlist[val1], linewidth=2)
                            leg.append('secondary {}'.format(val1))
                            alreadyDone = True
                        except AssertionError:
                            plt.plot(res['satW1'], res['capPres']/1000, linestyle=linelist[val1],
                                    color=colorlist[val1], linewidth=2, label='_nolegend_')
                            
                        res1=res['satW'].loc[res['capPres']<25000]
                        i = np.argmin(abs(res['satW'] - np.median(res1)))
                        x0, y0 = res['satW1'].iloc[i], res['capPres'].iloc[i]/1000
                        x1, y1 = res['satW1'].iloc[i+1], res['capPres'].iloc[i+1]/1000
                        plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.7", facecolor=colorlist[val1], mutation_scale=25), label='_nolegend_')
                        plt.annotate(self.description, xy=(0.3,12.0), label='_nolegend_',
                                        weight='bold', fontsize=14)
                    j += 1
                ind += 1
                j = 0
                
            else:
                res = self.results[val1]['pcSwDra']
                res1 = res.loc[res['source'] == 'MICP']
                res2 = res.loc[res['source'] != 'MICP']
                if not res1.empty:
                    plt.scatter(res1['satW'], res1['capPres']/1000, s=30, marker='o',
                                facecolors='none', edgecolors='k', linewidths=1.4, label='_nolegend_')
                    #leg.append('MICP')
                if not res2.empty:
                    plt.scatter(res2['satW'], res2['capPres']/1000, s=30, marker='s',
                                facecolors='none', edgecolors='k', linewidths=1.4, label='_nolegend_')
                    #from IPython import embed; embed()
                    #leg.append('Raeesi et al')

                res = self.results[val1]['pcSwImb']
                res1 = res.loc[res['source'] == 'Raeesi']
                res2 = res.loc[res['source'] == 'Lin']
                if not res1.empty:
                    plt.scatter(res1['satW'], res1['capPres']/1000, s=30, marker='s',
                                facecolors='none', edgecolors='k', label='_nolegend_',
                                linewidths=1.4)
                    #leg.append('Raeesi et al')
                if not res2.empty:
                    plt.scatter(res2['satW'], res2['capPres']/1000, s=30, marker='d',
                                facecolors='none', edgecolors='k', linewidths=1.4, label='_nolegend_')
                    #from IPython import embed; embed()
                    #leg.append('Lin et al')
            
        xlabel = r'$\mathbf{S_1}$'
        ylabel = r'Capillary Pressure ($\boldsymbol{kPa}$)'
        xlim = (0, 1.01)
        ylim = (0 , 25)
        self.formatFig(xlabel, ylabel, leg, xlim, ylim)
        plt.savefig(filename, dpi=500)
        plt.close()

    
    def krSw(self):
        filename = self.img_dir+'kr_vs_Sw_hysteresis_{}_{}_{}.png'.format(
            self.title, self.label, self.num)
        
        leg = []
        j = 0

        for val1 in self.results.keys():
            for val2 in self.results[val1].keys():
                res = self.results[val1][val2]
                print(res)
                plt.plot(res['satW'], res['krw'], self.linelist[j],
                        color=self.colorlist[j], linewidth=1.5)
                plt.plot(res['satW'], res['krnw'], self.linelist[j],
                        color=self.colorlist[j], linewidth=1.5, label = '_nolegend_')
                #leg.append(val)
                #from IPython import embed; embed()
                #leg.append(val1+'_'+val2)
                leg.append('cycle_{}'.format(j+1))
            j += 1
            

        xlabel = r'Sw'
        ylabel = r'Relative Permeability'
        xlim = (0, 1.01)
        ylim = (0 , 1.01)
        self.formatFig(xlabel, ylabel, leg, xlim, ylim)
        plt.savefig(filename, dpi=500)
        plt.close()

    def krSw1(self):
        filename = self.img_dir+'kr_vs_Sw_hysteresis_{}_{}_{}_wld.png'.format(
            self.title, self.label, self.num)
        
        leg = []
        j = 0

        from IPython import embed; embed()

        for val1 in self.results.keys():
            if val1 != 'Literature data':
                for val2 in self.results[val1].keys():
                    res = self.results[val1][val2]
                    print(res)                 
                    isFirstCycle = True
                    
                    try:
                        assert isFirstCycle
                        plt.plot(res['satW'], res['krw'], self.linelist[j],
                            color=self.colorlist[j], linewidth=2)
                        leg.append('cycle_{}'.format(j+1))
                    except AssertionError:
                        plt.plot(res['satW'], res['krw'], self.linelist[j],
                            color=self.colorlist[j], linewidth=2)
                        
                    plt.plot(res['satW'], res['krnw'], self.linelist[j],
                            color=self.colorlist[j], linewidth=2, label = '_nolegend_')

                    j += 1
                isFirstCycle = False

            else:
                res = self.results[val1]['krSwDra']        
                plt.scatter(res['satW'], res['krw'], s=30, marker='s',
                    facecolors='none', edgecolors='k', linewidths=1.4)
                leg.append('drainage Lit. data (krw)')
                plt.scatter(res['satW'], res['krnw'], s=30, marker='o',
                            facecolors='none', edgecolors='k', linewidths=1.4)
                leg.append('drainage Lit. data (krnw)')

                res = self.results[val1]['krSwImb']        
                plt.scatter(res['satW'], res['krw'], s=30, marker='s',
                    facecolors='none', edgecolors='r', linewidths=1.4)
                leg.append('imbibition Lit. data (krw)')
                plt.scatter(res['satW'], res['krnw'], s=30, marker='o',
                            facecolors='none', edgecolors='r', linewidths=1.4)
                leg.append('imbibition Lit. data (krnw)')
                
            
        xlabel = r'Sw'
        ylabel = r'Relative Permeability'
        xlim = (0, 1.01)
        ylim = (0 , 1.01)
        self.formatFig(xlabel, ylabel, leg, xlim, ylim)
        plt.savefig(filename, dpi=500)
        plt.close()


    def krSw2(self, process):
        filename = self.img_dir+'kr_vs_Sw_hysteresis_{}_{}_{}_{}_wld.png'.format(
            self.title, self.label, process, self.num)
        
        leg = []
        j = 0
        key = 'krSwDra' if process=='drainage' else 'krSwImb'
        loc = 9 if process=='drainage' else 1
        linelist = {'drainage': '--', 'imbibition': ':'}
        colorlist = {'drainage': 'r', 'imbibition': 'b'}
        isFirstCycle = True
        print('===========================================================================')
        print(process)

        for val1 in [process, 'Literature data']:
            alreadyDone = False
            if val1 != 'Literature data':
                for val2 in self.results[val1].keys():
                    res = self.results[val1][val2]
                    satW =(res['satW']+self.tuneParam)/(1+self.tuneParam)
                    minSw, maxSw = satW.min(), satW.max()
                    res = res.loc[(res['satW']>=minSw)&(res['satW']<=maxSw)]
                    #from IPython import embed; embed()
                    print(val1, val2)
                    
                    try:
                        assert isFirstCycle and val1=='drainage'
                        try:
                            assert self.includeTrapping
                            q, color = 0.65, 'k'
                            plt.plot(res['satW'], res['krw'], '-',
                                color='k', linewidth=2)
                            leg.append('primary {}'.format(val1))
                            isFirstCycle = False
                            plt.plot(res['satW'], res['krnw'], '-',
                                color='k', linewidth=2, label='_nolegend_')
                            

                            #res1=res['satW'].loc[res['capPres']<25000]
                            #from IPython import embed; embed()
                            #i = np.argmin(abs(res['satW'] - np.quantile(res1, q)))
                            sw = (res['satW'].max()-res['satW'].min())/2
                            i = np.argmin(np.abs(res['satW'] - sw))
                            x0, y0 = res['satW'].iloc[i], res['krw'].iloc[i]
                            x1, y1 = res['satW'].iloc[i+1], res['krw'].iloc[i+1]
                            plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                        arrowprops=dict(arrowstyle="-|>, head_length=0.7", edgecolor=color, facecolor=color, mutation_scale=25), label='_nolegend_')                                
                            
                            y0, y1 = res['krnw'].iloc[i], res['krnw'].iloc[i+1] 
                            plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                        arrowprops=dict(arrowstyle="-|>, head_length=0.7", edgecolor=color, facecolor=color, mutation_scale=25), label='_nolegend_')
                        except AssertionError:
                            pass
                        
                    except AssertionError:
                        print(val1, alreadyDone, linelist[val1])
                        q, color = 0.5, colorlist[val1]
                        try:
                            assert not alreadyDone
                            plt.plot(res['satW'], res['krw'], linelist[val1],
                                color=colorlist[val1], linewidth=2)
                            leg.append('secondary {}'.format(val1))
                            alreadyDone = True
                        except AssertionError:
                            plt.plot(res['satW'], res['krw'], linelist[val1],
                                color=colorlist[val1], linewidth=2, label='_nolegend_')
                            
                        plt.plot(res['satW'], res['krnw'], linelist[val1],
                            color=colorlist[val1], linewidth=2, label='_nolegend_')
                            
                        #res1=res['satW'].loc[res['capPres']<25000]
                        #from IPython import embed; embed()
                        #i = np.argmin(abs(res['satW'] - np.quantile(res1, q)))
                        sw = (res['satW'].max()-res['satW'].min())/2
                        i = np.argmin(np.abs(res['satW'] - sw))
                        x0, y0 = res['satW'].iloc[i], res['krw'].iloc[i]
                        x1, y1 = res['satW'].iloc[i+1], res['krw'].iloc[i+1]
                        plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.7", edgecolor=color, facecolor=color, mutation_scale=25), label='_nolegend_')                            
                        
                        y0, y1 = res['krnw'].iloc[i], res['krnw'].iloc[i+1] 
                        plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.7", edgecolor=color, facecolor=color, mutation_scale=25), label='_nolegend_')
                    
                    j += 1
                    isFirstCycle = False

                sw = (res['satW'].max()-res['satW'].min())/2
                i = np.argmin(np.abs(res['satW'] - sw))
                x0, y0 = res['satW'].iloc[i], res['capPres'].iloc[i]/1000
                x1, y1 = res['satW'].iloc[i+1], res['capPres'].iloc[i+1]/1000
                plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="-|>, head_length=0.7", facecolor=colorlist[val1], mutation_scale=25), label='_nolegend_')
                plt.annotate(self.description, xy=(0.3,0.7), label='_nolegend_',
                                weight='bold', fontsize=14)

            else:
                try:
                    assert self.includeTrapping
                    res = self.results[val1][key]        
                    plt.scatter(res['satW'], res['krw'], s=30, marker='s',
                        facecolors='none', edgecolors='k', linewidths=1.4)
                    #leg.append(r'Lit. data ($\mathbf{k_{r1}}$)')
                    plt.scatter(res['satW'], res['krnw'], s=30, marker='o',
                                facecolors='none', edgecolors='k', linewidths=1.4, label='_nolegend_')
                    #leg.append(r'Lit. data ($\mathbf{k_{r2}}$)')
                except AssertionError:
                    pass

            
        xlabel = r'$\mathbf{S_1}$'
        ylabel = r'Relative Permeability'
        xlim = (0, 1.01)
        ylim = (0 , 1.01)
        self.formatFig(xlabel, ylabel, leg, xlim, ylim, loc=loc)
        plt.savefig(filename, dpi=500)
        plt.close()


    def krSw3(self):
        filename = self.img_dir+'kr_vs_Sw_hysteresis_{}_{}.png'.format(
            self.title, self.label)
        
        leg = []
        j = 0
        linelist = {'drainage': '-', 'imbibition': '-.'}
        colorlist = {'drainage': 'r', 'imbibition': 'b'}
        markerlist = {'drainage': 's', 'imbibition': 'o'}
        isFirstCycle = True
        print('===========================================================================')

        for val1 in ['drainage', 'imbibition']:
            #from IPython import embed; embed()
            alreadyDone = False
            for val2 in self.results[val1].keys():
                res = self.results[val1][val2]
                satW =(res['satW']+self.tuneParam)/(1+self.tuneParam)
                minSw, maxSw = satW.min(), satW.max()
                res = res.loc[(res['satW']>=minSw)&(res['satW']<=maxSw)]
                
                try:
                    assert isFirstCycle and val1=='drainage'
                    try:
                        assert self.includeTrapping
                        
                        plt.plot(res['satW'], res['krw'], '-',
                            color='k', linewidth=2)
                        leg.append('primary {}'.format(val1))
                        isFirstCycle = False
                        plt.plot(res['satW'], res['krnw'], '-',
                            color='k', linewidth=2, label='_nolegend_')
                        q, color = 0.25, 'k'

                        #res1=res['satW'].loc[res['capPres']<25000]
                        #from IPython import embed; embed()
                        sw = (res['satW'].max()-res['satW'].min())/2
                        i = np.argmin(np.abs(res['satW'] - sw))
                        x0, y0 = res['satW'].iloc[i], res['krw'].iloc[i]
                        x1, y1 = res['satW'].iloc[i+1], res['krw'].iloc[i+1]
                        plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.7", facecolor=color, mutation_scale=25), label='_nolegend_')                            
                        
                        y0, y1 = res['krnw'].iloc[i], res['krnw'].iloc[i+1] 
                        plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                    arrowprops=dict(arrowstyle="-|>, head_length=0.7", facecolor=color, mutation_scale=25), label='_nolegend_')
                    except AssertionError:
                        pass
                    
                except AssertionError:
                    #print(val1, alreadyDone, linelist[val1])
                    q, color = 0.75, colorlist[val1]
                    try:
                        assert not alreadyDone
                        plt.plot(res['satW'], res['krw'], linelist[val1], 
                                 #marker=markerlist[val1], markevery=0.1, 
                                 color=colorlist[val1], linewidth=2)
                        leg.append('secondary {}'.format(val1))
                        alreadyDone = True
                    except AssertionError:
                        plt.plot(res['satW'], res['krw'], linelist[val1], 
                                 #marker=markerlist[val1], markevery=0.1, 
                                 color=colorlist[val1], linewidth=2, label='_nolegend_')
                        
                    plt.plot(res['satW'], res['krnw'], linelist[val1], 
                             #marker=markerlist[val1], markevery=0.1, 
                             color=colorlist[val1], linewidth=2, label='_nolegend_')
                    
                    sw = (res['satW'].max()-res['satW'].min())/2
                    i = np.argmin(np.abs(res['satW'] - sw))
                    x0, y0 = res['satW'].iloc[i], res['krw'].iloc[i]
                    x1, y1 = res['satW'].iloc[i+1], res['krw'].iloc[i+1]
                    plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                arrowprops=dict(arrowstyle="-|>, head_length=0.7", facecolor=color, edgecolor=color, mutation_scale=25), label='_nolegend_')
                    
                    y0, y1 = res['krnw'].iloc[i], res['krnw'].iloc[i+1] 
                    plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                arrowprops=dict(arrowstyle="-|>, head_length=0.7", facecolor=color, edgecolor=color, mutation_scale=25), label='_nolegend_')
                
                j += 1
                isFirstCycle = False

            i = np.argmin(np.abs(res['krw'] - np.median(res['krw'])))
            x0, y0 = res['satW'].iloc[i], res['capPres'].iloc[i]/1000
            x1, y1 = res['satW'].iloc[i+1], res['capPres'].iloc[i+1]/1000
            plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>, head_length=0.7", facecolor=colorlist[val1], mutation_scale=25), label='_nolegend_')
            plt.annotate(self.description, xy=(0.3,0.7), label='_nolegend_',
                            weight='bold', fontsize=14)

            
        xlabel = r'$\mathbf{S_1}$'
        ylabel = r'Relative Permeability'
        xlim = (0, 1.01)
        ylim = (0 , 1.01)
        self.formatFig(xlabel, ylabel, leg, xlim, ylim)
        plt.savefig(filename, dpi=500)
        plt.close()


    def krSwScaled(self):
        fluid = 'nw'
        Sw = 'Se'
        fluidlabel = 1 if fluid=='w' else 2
        Swlabel = 'Se' if Sw=='Se' else 'satW2' if Sw=='S2' else 'satW'
        log = False
        nameLabel = '_'
        if Sw=='Se': nameLabel += 'scaled_'
        if log: nameLabel += 'log_'

        filename = self.img_dir+'kr_vs_{}_hysteresis_{}{}{}.png'.format(
            Sw, self.title, nameLabel, fluid)
        
        leg = []
        linelist = {'drainage': {'nt':'-.', 'wt':'-'}, 'imbibition': {'nt':'-.', 'wt':':'}}
        colorlist = {'drainage': {'wt':'b', 'nt':'r'}, 
                     'imbibition': {'wt':'brown', 'nt':'k'}
                    }
        markerlist = {'drainage': {'wt':'s', 'nt':'o'}, 'imbibition': {'wt':'X', 'nt':'v'}}
        markeverylist = [0.01, 0.07, 0.09, 0.4]
        
        q=0.4
        ind, indi, j = 0, 0, 0
        for label in ['wt', 'nt']:
            isFirstCycle = True

            res = self.results['imbibition'][label+'_cycle1']
            label1 = 'trapping' if label=='wt' else 'no trapping'
            for val1 in ['drainage', 'imbibition']:
                isFirstCycle = True
                for val2 in [label+'_cycle1', label+'_cycle2', label+'_cycle3']:
                    try:
                        assert j!=0 or val1=='imbibition'
                        res = self.results[val1][val2]
                        satW =(res['satW']+self.tuneParam)/(1+self.tuneParam)
                        minSw, maxSw = satW.min(), satW.max()
                        res = res.loc[(res['satW']>=minSw)&(res['satW']<=maxSw)]

                        Swi = res['satW'].min()
                        Sgr = 1 - res['satW'].max()
                        res['Se'] = (res['satW']-Swi)/(1-Swi-Sgr)
                        res['satW2'] = 1-res['satW']
                        
                        try:
                            assert isFirstCycle
                            plt.plot(res[Swlabel], res['kr'+fluid], 
                                     linestyle=linelist[val1][label],
                                     color=colorlist[val1][label], 
                                     marker=markerlist[val1][label], 
                                     markersize=8, #markevery=markeverylist[j],
                                     linewidth=3)
                            leg.append('{} ({})'.format(val1, label1))
                            isFirstCycle = False
                        except AssertionError:
                            plt.plot(res[Swlabel], res['kr'+fluid], linestyle=linelist[val1][label],
                                color=colorlist[val1][label], linewidth=3, label = '_nolegend_')
                    except AssertionError:
                        pass
                
                    j += 1
                    ind += 1
                j, ind = 0, indi
                isFirstCycle = False
            indi = 3
            ind = indi
            
        #xlabel = r'Se'
        xlabel_list={'Se': r'$\mathbf{S_e}$', 'satW': r'$\mathbf{S_1}$', 
                     'satW2': r'$\mathbf{S_2}$'}
        xlabel = xlabel_list[Swlabel]
        ylabel = r'Relative Permeability (phase {})'.format(fluidlabel)
        xlim = (0, 1.01)
        ylim = (0 , 1.01)
        if log: plt.yscale('log')
        loc=0
        self.formatFig(xlabel, ylabel, leg, xlim, ylim, loc=loc)
        plt.savefig(filename, dpi=500)
        plt.close()

    
    def krSwProposed(self):
        Sw = 'Sw'
        Swlabel = 'Se' if Sw=='Se' else 'satW'
        filename = self.img_dir+'kr_vs_{}_hysteresis_{}_proposed.png'.format(
            Sw, self.title)
        
        leg = []
        linelist = {'drainage': {'nt':'-.', 'wt':'-'}, 'imbibition': {'nt':'-.', 'wt':':'}}
        colorlist = {'drainage': {'wt':'b', 'nt':'r'}, 
                     'imbibition': {'wt':'brown', 'nt':'k'}
                    }
        markerlist = {'drainage': {'wt':'s', 'nt':'o'}, 'imbibition': {'wt':'X', 'nt':'v'}}
        markeverylist = [0.01, 0.07, 0.09, 0.4]

        q=0.4
        ind, indi, j = 0, 0, 0
        label='wt'
        #for label in ['wt','nt']:
        isFirstCycle = True
        label1 = 'trapping' if label=='wt' else 'no trapping'

        #from IPython import embed; embed()
        resI_t = self.results['imbibition']['wt_cycle3']
        S1t_t = resI_t['satW'].max()
        kr1_It_S1t_t = resI_t['krw'].max()

        resD = self.results['drainage']['wt_cycle1']
        kr1_Dt_S1t_t = np.interp(x=S1t_t, 
                                 xp=resD['satW'].values[::-1],
                                 fp=resD['krw'].values[::-1])
        S1t_n = self.results['imbibition']['nt_cycle3']['satW'].max()

        satW =(resD['satW']+self.tuneParam)/(1+self.tuneParam)
        minSw, maxSw = satW.min(), satW.max()
        resD = resD.loc[(resD['satW']>=minSw)&(resD['satW']<=maxSw)]
        print(resD)
        plt.plot(resD['satW'], resD['krw'], linestyle='-.',
                 color='c', linewidth=3,
                 marker='o', markersize=8)
        leg.append('primary drainage')
        
        satW =(resI_t['satW']+self.tuneParam)/(1+self.tuneParam)
        minSw, maxSw = satW.min(), satW.max()
        resI_t = resI_t.loc[(resI_t['satW']>=minSw)&(resI_t['satW']<=maxSw)]
        print(resI_t)
        plt.plot(resI_t['satW'], resI_t['krw'], linestyle='-',
                 color='b', linewidth=3,
                 marker='v', 
                 markersize=8)
        leg.append('imbibition (with trapping)')

        resI_n = self.results['imbibition']['nt_cycle3']
        satW =(resI_n['satW']+self.tuneParam)/(1+self.tuneParam)
        minSw, maxSw = satW.min(), satW.max()
        resI_n = resI_n.loc[(resI_n['satW']>=minSw)&(resI_n['satW']<=maxSw)]
        print(resI_n)
        plt.plot(resI_n['satW'], resI_n['krw'], linestyle='--',
                 color='k', linewidth=3,
                 #marker='x', markersize=8
                 )
        leg.append('imbibition (no trapping)')

        x = np.linspace(S1t_t, S1t_n, 10)
        y = np.interp(x=x, 
                  xp=resD['satW'].values[::-1], 
                  fp=resD['krw'].values[::-1])
        x = np.append(resI_t['satW'].values, x)
        y = np.append(resI_t['krw'].values, y*kr1_It_S1t_t/kr1_Dt_S1t_t)
        print(np.array([*zip(x,y)]))
        plt.plot(x, y, linestyle=':',
                 color='r', linewidth=3)
        leg.append('predicted, Eq. (2)')

        #xlabel = r'Se'
        xlabel = r'$\mathbf{S_1}$' if Sw!='Se' else r'$\mathbf{S_e}$'
        ylabel = r'Relative Permeability (phase 1)'
        xlim = (0, 1.01)
        ylim = (0 , 1.01)
        #plt.yscale('log')
        loc=0
        self.formatFig(xlabel, ylabel, leg, xlim, ylim, loc=loc)
        plt.savefig(filename, dpi=500)
        plt.close()


    def plotDistribution(self):
        filename = self.img_dir+'freq_dist_hysteresis_{}.png'.format(
            self.title)
        
        leg = []
        #ind, indi, j = 0, 0, 0
        ymax=0
        bin=15
        val1 = 'imbibition'
        linelist = {'wt':'dashdot', 'nt':':', 'pores':'-', 'throats':'--'}
        colorlist = {'wt':'b', 'nt':'r'}
        alreadyDone = True

        for label in ['wt', 'nt']:
            val2 = label+'_cycle1_trDi'
            label1 = 'trapping' if label=='wt' else 'no trapping'
            res = self.results[val1][val2]
            for elem in ['pores', 'throats']:
                res1 = res.iloc[1:8223] if elem=='pores' else res.iloc[8223:-1]
                label2 = label1 if elem=='pores' else '_nolegend_'
                #from IPython import embed; embed()
                if elem=='pores': leg.append(label1)
                res1[['trappedW','trappedNW']] = res1[['trappedW','trappedNW']].astype('bool')
                xx,yy=self.__plotDistribution__(
                    res1['rad'], 
                    res1['volume'],
                    #np.ones(len(res1['fluid'])),
                    #res1['fluid'],
                    res1['trappedW']|res1['trappedNW'],
                    colorlist[label], 2, 10, 
                    lineStyle=linelist[label],
                    pltLabel=label2)
    
                try:
                    assert not alreadyDone
                    xx,yy=self.__plotDistribution__(
                        res1['rad'], 
                        res1['volume'],
                        np.ones(len(res1['fluid'])),
                        'k', 2, bin, lineStyle=linelist[elem],
                        pltLabel=elem)
                    leg.append('all '+elem)
                    ymax=max(ymax,max(yy))
                except AssertionError:
                    pass
            
            alreadyDone = False

        xlabel = r'radius (microns)'
        ylabel = r'Frequency'
        xlim = (0, None)
        ylim = (0 , ymax)
        self.formatFig(xlabel, ylabel, leg, xlim, ylim)
        plt.savefig(filename, dpi=500)
        plt.close()       

    
    def __plotDistribution__(self, radiusData, weights, fFaz, color, lineWidth, n, 
                             pltLabel='_nolegend_', lineStyle='-', markerStyle=None):
    
        vTotal = np.sum(weights)
        rMax = np.max(radiusData) + 1e-32
        rMin = np.min(radiusData) - 1e-32
        drV = (rMax - rMin) / (n - 1)
        Edges = np.zeros(shape=(n, 1), dtype=float)
    
        fr = np.zeros(shape=(n - 1, 1), dtype=float)
        Edges[0] = rMin
        
        tempArray = (radiusData * fFaz)
        for ii in range(n - 1):        
            rUpper = rMin + (ii + 1) * drV
            rLower = rMin + (ii) * drV
            Edges[ii + 1] = rUpper
            if ii == n - 2:
                rUpper = np.inf

            temp = ((tempArray >= rLower) & (tempArray < rUpper))
        
            fr[ii] = np.sum(weights[temp])
        
        rV = Edges[:n - 1] + np.diff(Edges, axis=0) / 2
        minrV = min(rV)

        rV = np.insert(rV, 0, 0.0)
        fr = np.insert(fr, 0, 0.0)
        rV = np.append(rV, rV[-1] + drV)
        fr = np.append(fr, 0.0)

        xx = rV * 1e6
        yy = fr / (vTotal)

        print(xx, yy)

        if minrV * 1e6 > 10:
            xx[0] = minrV * 1e6 - 0.00001

        plt.plot(xx, yy, color=color, linestyle=lineStyle, linewidth=lineWidth, 
                 marker=markerStyle, markersize=6, markerfacecolor='None', 
                 alpha=1.0, label=pltLabel)
        '''plt.fill_between(x= xx, y1= yy, 
                         #where= (-1 < t)&(t < 1),
                         color=color,
                         alpha= 0.1, label=pltLabel)'''

        plt.savefig('ddddddddd.png', dpi=500)

        return xx, yy


class drainageBank:
    def __init__(self, obj):
        self.obj = obj
        if self.compWithLitData:
            self.__compWithLitData__()
        if self.compWithPrevData:
            self.__compWithPrevData__()

    def __getattr__(self, name):
        return getattr(self.obj, name)

    def __compWithLitData__(self):
        self.results['Literature data'] = self.results.get('Literature data', {})
        
        self.results['Literature data']['pcSwDra'] = pd.read_csv(
            './results_csv/Exp_Results_Bentheimer_Drainage_Pc_Sw.csv',
            names=['source', 'satW', 'Pc', 'capPres'], sep=',',
            skiprows=1, index_col=False)
        
        self.results['Literature data']['krSwDra'] = pd.read_csv(
            './results_csv/Exp_Results_Bentheimer_Drainage_kr_Sw.csv',
            names=['satW', 'krw', 'krnw'], sep=',',
            skiprows=1, index_col=False)
        
        
        '''self.results['Valvatne et al.'] = pd.read_csv(
            './results_csv/pnflow_Bentheimer_Drainage_010725.csv',
            names=['satW', 'capPres', 'krw', 'krnw', 'RI'], sep=',',
            skiprows=1, index_col=False)'''
        
    def __compWithPrevData__(self):
        if self.include:
            todo = list(self.include)
        else:
            todo = np.arange(1, self.num).tolist()
            if self.exclude:
                todo = np.setdiff1d(todo, self.exclude).tolist()

        while True:
            try:
                n = todo.pop(0)
                self.results['model_'+str(n)] = pd.read_csv(
                    "./results_csv/FlowmodelOOP_{}_Drainage_{}.csv".format(self.title, n),
                    names=['satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
                    sep=',', skiprows=18, index_col=False)
            except FileNotFoundError:
                pass
            except IndexError:
                break


class imbibitionBank():
    def __init__(self, obj):
        self.obj = obj
        if self.compWithLitData:
            self.__compWithLitData__()
        if self.compWithPrevData:
            self.__compWithPrevData__()

    def __getattr__(self, name):
        return getattr(self.obj, name)
        
    def __compWithLitData__(self):
        self.results['Literature data'] = self.results.get('Literature data', {})
        
        self.results['Literature data']['pcSwImb'] = pd.read_csv(
            './results_csv/Exp_Results_Bentheimer_Imbibition_Pc_Sw.csv',
            names=['source', 'satW', 'Pc', 'capPres', 'scaledPc'], sep=',',
            skiprows=1, index_col=False)
        self.results['Literature data']['krSwImb'] = pd.read_csv(
            './results_csv/Exp_Results_Bentheimer_Imbibition_kr_Sw.csv',
            names=['satW', 'krw', 'krnw'], sep=',',
            skiprows=1, index_col=False)
        
        '''self.results['Valvatne et al.'] = pd.read_csv(
            './results_csv/pnflow_Bentheimer_Imbibition_010725.csv', names=[
                'satW', 'capPres', 'krw', 'krnw', 'RI'], sep=',', skiprows=1,
            index_col=False)'''
            
    def __compWithPrevData__(self):
        if self.include:
            todo = list(self.include)
        else:
            todo = np.arange(1, self.num).tolist()
            if self.exclude:
                todo = np.setdiff1d(todo, self.exclude).tolist()
        while True:
            try:
                n = todo.pop(0)
                self.results['model_'+str(n)] = pd.read_csv(
                    "./results_csv/FlowmodelOOP_{}_Imbibition_{}.csv".format(self.title, n),
                    names=['satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
                    sep=',', skiprows=18, index_col=False)
            except FileNotFoundError:
                pass
            except IndexError:
                break