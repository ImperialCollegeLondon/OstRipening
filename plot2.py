import pandas as pd
import numpy as np
from matplotlib import markers, axes, pyplot as plt
import os

#4_1
img_dir = "./result_images/"
os.makedirs(os.path.dirname(img_dir), exist_ok=True)
num = 98
'''data = pd.read_csv('results_csv/Flowmodel_Bentheimer_ostRipening_4_1.csv', 
    sep=',', skiprows=1, names=[
    'step', 'totalFlux', 'total_delta_nMoles', 'Sw', '#cluster_growth',
    '#cluster_shrinkage', 'AvgPressure'])'''
data = pd.read_csv('results_csv/Flowmodel_Bentheimer_ostRipening_{}.csv'.format(num), 
    sep=',', skiprows=1, names=[
    'step', 'totalFlux', 'total_delta_nMoles', 'Sw', '#cluster_growth',
    '#cluster_shrinkage', 'AvgPressure', 'totMolesInW', 'totMolesInNW'],
    index_col=False)

#print((data['#cluster_shrinkage']>100).sum(), data['#cluster_shrinkage'].max(),
 #     data['#cluster_shrinkage'].sort_values(ascending=False))


def formatFig(xlabel, ylabel, leg, xlim, ylim, loc=1):
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


def _formatFig(axList, legend):
        label_font = {'fontname': 'Arial', 'size': 14, 'color': 'black', 'weight': 'bold',
                      'labelpad': 10}
        
        for ax in axList:
            ax.tick_params(direction='in', axis='both', which='major', pad=10)
            
            # Increase font size and make it bold        
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() 
                        + ax.get_yticklabels()+ legend.get_texts()):
                item.set_fontsize(14)
                item.set_fontname('Arial')
                item.set_fontweight('bold')
    
        plt.tight_layout()



def plotSaturation():
    filename = img_dir+'Bentheimer_ostRip_saturation_{}.png'.format(num)

    fig, ax1 = plt.subplots(figsize=(36,6))

    # Plotting y1 with the primary y-axis
    line1, = ax1.plot(data['step'], data['Sw'], 'b-', label=r'$\mathbf{S_1}$')
    ax1.set_xlabel('time steps')
    #ax1.set_xlim((-2, 1000))
    ax1.set_ylabel('Saturation', color='b')
    #ax1.set_ylim((0.6, 0.7))
    ax1.tick_params(axis='y', labelcolor='b')

    # Creating a secondary y-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(data['step'], data['#cluster_growth'], 'r-', label='cluster_growth')
    line3, = ax2.plot(data['step'], data['#cluster_shrinkage'], 'k-', label='cluster_shrinkage')
    ax2.yaxis.get_major_locator().set_params(integer=True)
    ax2.set_ylabel('No of cluster events', color='r')
    #ax2.set_ylim((0, 50))
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine legends from both axes
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    legend = ax1.legend(lines, labels, loc='center', bbox_to_anchor=(0.9, 0.9))

    _formatFig([ax1, ax2], legend)
    plt.text(500, 45, 'each time step = 1.0s', fontsize=14, fontweight='bold')
    plt.savefig(filename, dpi=500)
    plt.close()


def plotPressure():
    filename = img_dir+'Bentheimer_ostRip_pressure_{}.png'.format(num)

    fig, ax1 = plt.subplots(figsize=(36,6))

    # Plotting y1 with the primary y-axis
    line1, = ax1.plot(data['step'], data['Sw'], 'b-', label=r'$\mathbf{S_1}$')
    ax1.set_xlabel('time steps')
    #ax1.set_xlim((-2, 1000))
    ax1.set_ylabel('Saturation', color='b')
    ax1.set_ylim((0.6, 0.7))
    ax1.tick_params(axis='y', labelcolor='b')

    # Creating a secondary y-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(data['step'], data['AvgPressure'], 'r-', label='Average Pressure')
    #line3, = ax2.plot(data['step'], data['#cluster_shrinkage'], 'k-', label='cluster_shrinkage')
    ax2.yaxis.get_major_locator().set_params(integer=True)
    ax2.set_ylabel('Average Pressure (Pa)', color='r')
    ax2.set_ylim((1200, 2250))
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine legends from both axes
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    legend = ax1.legend(lines, labels, loc='center', bbox_to_anchor=(0.9, 0.9))

    _formatFig([ax1, ax2], legend)
    plt.text(500, 2200, 'each time step = 1.0s', fontsize=14, fontweight='bold')
    plt.savefig(filename, dpi=500)
    plt.close()


def plotTotalValues():
    filename = img_dir+'Bentheimer_ostRip_total_values_{}.png'.format(num)

    fig, ax1 = plt.subplots(figsize=(36,6))

    # Plotting y1 with the primary y-axis
    line1, = ax1.plot(data['step'], data['totalFlux'], 'b-', label='Total flux')
    ax1.set_xlabel('time steps')
    #ax1.set_xlim((-2, 1000))
    ax1.set_ylabel('Total flux (mol/(m2.s))', color='b')
    ax1.set_ylim((-1e-17, 1e-17))
    ax1.tick_params(axis='y', labelcolor='b')

    # Creating a secondary y-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(data['step'], data['total_delta_nMoles'], 'k-', label='Total change of moles')
    #line3, = ax2.plot(data['step'], data['total_delta_nMoles'], 'k-', label='Total change of moles')
    #ax2.yaxis.get_major_locator().set_params(integer=True)
    ax2.set_ylabel("Total change of moles ('mol')", color='r')
    ax2.set_ylim((-1e-25, 1e-25))
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine legends from both axes
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    legend = ax1.legend(lines, labels, loc='center', bbox_to_anchor=(0.9, 0.9))

    _formatFig([ax1, ax2], legend)
    plt.text(500, 9e-26, 'each time step = 0.1s', fontsize=14, fontweight='bold')
    plt.savefig(filename, dpi=500)
    plt.close()



plotSaturation()
plotTotalValues()
plotPressure()