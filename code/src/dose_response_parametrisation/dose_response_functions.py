import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

def load_dose_response(filename,HSL_transform =0.14*10**3):
    data = pkl.load(open('../../input/dose_response_data/%s'%filename,'rb'))
    OC14_list= np.array(data['OC14'])*HSL_transform; gfpExp_list = list(data['mean_gfp']); rfpExp_list = list(data['mean_rfp'])
    semGreen = list(data['std_gfp']); semRed = list(data['std_rfp'])
    return OC14_list, gfpExp_list, rfpExp_list, semGreen, semRed


def plotData(inducer, rfpExp_list, gfpExp_list, semRed, semGreen,pad=0.01, inducerName='HSL'):
    fig,ax = plt.subplots()

    ax.plot(inducer,rfpExp_list,label='RFP', c='red')
    ax.scatter(inducer,rfpExp_list, c='red')
    ax.errorbar(inducer,rfpExp_list,yerr=semRed,c='red',fmt='o')
    ax.legend(loc='center left') #upper right
    ax.set_ylabel('RFP / ($A_{600}$ $RFP_{basal})$')
    ax.set_xscale('log')


    ax2=ax.twinx()
    ax2.plot(inducer,gfpExp_list,label='GFP', c='green')
    ax2.scatter(inducer,gfpExp_list, c='green')
    ax2.errorbar(inducer,gfpExp_list,yerr=semGreen,c='green',fmt='o')
    ax2.legend(loc='center right') #upper left
    ax2.set_ylabel('GFP / ($A_{600}$ $GFP_{basal})$')
    ax.set_xscale('log')
    ax.set_xlabel(f'{inducerName} concentration (dimensionless)')
    plt.show()

def plotFitvsData(inducer,inducer_continuous, gfpExp_list, rfpExp_list, semGreen, semRed,doseResponseGreen,doseResponseRed,pad=0.01, inducerName='HSL'):
    fig,ax = plt.subplots()

    ax.plot(inducer_continuous,doseResponseRed,label='RFP', c='red')
    ax.scatter(inducer,rfpExp_list, c='red')
    ax.errorbar(inducer,rfpExp_list,yerr=semRed,c='red',fmt='o')
    ax.legend(loc='center left') #upper right
    ax.set_ylabel('RFP / ($A_{600}$ $RFP_{basal})$')
    ax.set_xscale('log')


    ax2=ax.twinx()
    ax2.plot(inducer_continuous,doseResponseGreen,label='GFP', c='green')
    ax2.scatter(inducer,gfpExp_list, c='green')
    ax2.errorbar(inducer,gfpExp_list,yerr=semGreen,c='green',fmt='o')
    ax2.legend(loc='center right') #upper left
    ax2.set_ylabel('GFP / ($A_{600}$ $GFP_{basal})$')
    ax.set_xscale('log')
    ax.set_xlabel(f'{inducerName} concentration (dimensionless)')
    plt.show()




def plotData_croppedGreen(inducerGreen, inducerRed, rfpExp_list, gfpExp_list, semRed, semGreen,pad=0.01, inducerName='HSL'):
    fig,ax = plt.subplots()

    ax.plot(inducerRed,rfpExp_list,label='RFP', c='red')
    ax.scatter(inducerRed,rfpExp_list, c='red')
    ax.errorbar(inducerRed,rfpExp_list,yerr=semRed,c='red',fmt='o')
    ax.legend(loc='center left') #upper right
    ax.set_ylabel('RFP / ($A_{600}$ $RFP_{basal})$')
    ax.set_xscale('log')


    ax2=ax.twinx()
    ax2.plot(inducerGreen,gfpExp_list,label='GFP', c='green')
    ax2.scatter(inducerGreen,gfpExp_list, c='green')
    ax2.errorbar(inducerGreen,gfpExp_list,yerr=semGreen,c='green',fmt='o')
    ax2.legend(loc='center right') #upper left
    ax2.set_ylabel('GFP / ($A_{600}$ $GFP_{basal})$')
    ax.set_xscale('log')
    ax.set_xlabel(f'{inducerName} concentration (dimensionless)')

    plt.show()

def plotFitvsData_croppedGreen(inducer_green, inducer_red,inducer_continuous, gfpExp_list, rfpExp_list, semGreen, semRed,doseResponseGreen,doseResponseRed,pad=0.01, inducerName='HSL',save_figure=False, save_figure_name='dose_response.pdf'):
    fig,ax = plt.subplots()

    ax.plot(inducer_continuous,doseResponseRed,label='RFP', c='red')
    ax.scatter(inducer_red,rfpExp_list, c='red')
    ax.errorbar(inducer_red,rfpExp_list,yerr=semRed,c='red',fmt='o')
    ax.legend(loc='center left') #upper right
    ax.set_ylabel('RFP / ($A_{600}$ $RFP_{basal})$')
    ax.set_xscale('log')


    ax2=ax.twinx()
    ax2.plot(inducer_continuous,doseResponseGreen,label='GFP', c='green')
    ax2.scatter(inducer_green,gfpExp_list, c='green')
    ax2.errorbar(inducer_green,gfpExp_list,yerr=semGreen,c='green',fmt='o')
    ax2.legend(loc='center right') #upper left
    ax2.set_ylabel('GFP / ($A_{600}$ $GFP_{basal})$')
    ax.set_xscale('log')
    ax.set_xlabel(f'{inducerName} concentration (dimensionless)')
    if save_figure==True:

        plt.savefig(save_figure_name, bbox_inches='tight')

    plt.show()



def plot_multiple_dose_response(OC14_list1_green,OC14_list1_red,gfpExp_list1, rfpExp_list1, gfpExp_list3,rfpExp_list3,semGreen1, semRed1, semGreen3, semRed3, popt_subcircuit1, popt_subcircuit3):
    fig,ax = plt.subplots()
    ax2=ax.twinx()
    OC14_continuous = np.logspace(-3,2, 100)*HSLtransform
    best_fit_p = np.hstack([popt_subcircuit1, popt_subcircuit3])
    for p in filtered_parameters:
        fluorescenceFit = steadystate_combined_subcircuits(OC14_continuous, *p)
        fluorescenceFit_continuous = steadystate_combined_subcircuits(OC14_continuous, *p)
        fluorescenceSingleFit_continuous = steadystate_combined_subcircuits(OC14_continuous, *best_fit_p)

        
        ax2.plot(OC14_continuous, fluorescenceFit_continuous[0], c='darkseagreen', alpha=0.08)
        ax.plot(OC14_continuous, fluorescenceFit_continuous[1], c='lightcoral', alpha=0.08)
        ax2.scatter(OC14_list1_green,gfpExp_list1 , label='data', c='green')
        ax2.errorbar(OC14_list1_green,gfpExp_list1,yerr=semGreen1,c='green',fmt='o')
        ax.scatter(OC14_list1_red,rfpExp_list1 , label='data', c='red')
        ax.errorbar(OC14_list1_red,rfpExp_list1,yerr=semRed1,c='red',fmt='o')
        plt.xscale('log')


    # ax.legend(loc='center left') #upper right
    ax.set_ylabel('RFP / ($A_{600}$ $RFP_{basal})$', fontsize=15)
    ax.set_xscale('log')
    # ax2.legend(loc='center right') #upper left
    ax2.set_ylabel('GFP / ($A_{600}$ $GFP_{basal})$',fontsize=15)
    ax.set_xscale('log')
    ax.set_xlabel(f'3OHC14-HSL concentration (µM)',fontsize=15)
    ax.plot(OC14_continuous, rfpFit1_continuous_copy, c='red', alpha=1)
    ax2.plot(OC14_continuous, gfpFit1_continuous_copy, c='green', alpha=1)
    plt.show()

    gfpFit1_continuous_copy
    fig,ax = plt.subplots()
    ax2=ax.twinx()
    for p in filtered_parameters:
        fluorescenceFit = steadystate_combined_subcircuits(OC14_continuous, *p)
        fluorescenceFit_continuous = steadystate_combined_subcircuits(OC14_continuous, *p)
        fluorescenceSingleFit_continuous = steadystate_combined_subcircuits(OC14_continuous, *best_fit_p)

        
        ax2.plot(OC14_continuous, fluorescenceFit_continuous[2], c='darkseagreen', alpha=0.08)
        ax.plot(OC14_continuous, fluorescenceFit_continuous[3], c='lightcoral', alpha=0.08)
        ax2.scatter(OC14_list1_green,gfpExp_list3 , label='data', c='green')
        ax2.errorbar(OC14_list1_green,gfpExp_list3,yerr=semGreen3,c='green',fmt='o')
        ax.scatter(OC14_list1_red,rfpExp_list3 , label='data', c='red')
        ax.errorbar(OC14_list1_red,rfpExp_list3,yerr=semRed3,c='red',fmt='o')
        plt.xscale('log')

    # ax.legend(loc='center left') #upper right
    ax.set_ylabel('RFP / ($A_{600}$ $RFP_{basal})$', fontsize=15)
    ax.set_xscale('log')
    # ax2.legend(loc='center right') #upper left
    ax2.set_ylabel('GFP / ($A_{600}$ $GFP_{basal})$', fontsize=15)
    ax.set_xscale('log')
    ax.set_xlabel(f'3OHC14-HSL concentration (µM)', fontsize=15)

    ax.plot(OC14_continuous, rfpFit3_continuous_copy, c='red', alpha=1)
    ax2.plot(OC14_continuous, gfpFit3_continuous_copy, c='green', alpha=1)
    plt.show()