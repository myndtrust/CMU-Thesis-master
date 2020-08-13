#!/usr/bin/python3

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from geopandas import GeoDataFrame
from shapely.geometry import Point
import geopandas as gpd
import networkx as nx
from collections import Counter
import itertools
from statsmodels.tsa.arima_model import ARIMA
import warnings
from fbprophet import Prophet as fbp
from fbprophet.plot import add_changepoints_to_plot
from termcolor import colored

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() # This toggles Eager mode
# import tensorflow.keras.layers as layers
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# import tensorflow.keras.backend as K
# from tensorflow.keras.initializers import glorot_uniform
# from termcolor import colored
# from tensorflow.python.client import device_lib;
# print(colored("TF Version: "+str(tf.__version__)+" GPU Available: "+str(tf.test.is_gpu_available()), 'red'))
# print(colored("Eager on: "+str(tf.executing_eagerly()),'red'))

# print(colored('Imported Modules\n', 'yellow'))
# print(colored('Running from '+str((os.getcwd())),'green'))
# print(colored('Other directories at this level are '+ str(os.listdir()),'red'))

# Set path to data sources
key_1 = 'key_1.csv'
key_2 = 'key_2.csv'
train_1 = 'train_1.csv'
train_2 = 'train_2.csv'
utest = 'unit_testing.csv'
path = '../../Manuscript/images/'


class traffic(object):
    def __init__(self, train, quantile, is_test=True):
        '''Args: train: times serires data-set to train on.
                 quanitile: ['mean'| float], where float is 0<1, ie 0.95'''
        self.root = '/home/eric/ramukcire/estimating_cost_of_dc_services/syscost/traffic/data/'
        self.train = self.root + train
        self.is_test = is_test
        if self.is_test != True:
            self.df = pd.read_csv(self.train).fillna(0)
            for col in self.df.columns[1:]:
                self.df[col] = pd.to_numeric(self.df[col],downcast='integer')
        else:
            self.df = pd.read_csv(self.train).fillna(0)

        self.quantile = quantile

        self.langs = ['en','ja','de','fr','zh','ru','es']

        self.labels = {'en':'English', 'ja':'Japanese', 'de':'German',
                       'fr':'French', 'zh':'Chinese', 'ru':'Russian','es':'Spanish'}

        self.serving = pd.read_csv(self.root+'country_ingress.csv', index_col = 0)

        self.dcs = self.serving['ingress'].unique().tolist()

        self.df['lang'] = self.df.Page.map(self.get_language)

        self.lps = pd.read_csv(self.root+'lang_2_pg_size.csv', index_col = 0)

        self.lang_sets = {}
        for l in self.langs:
            self.lang_sets[l] = self.df[self.df['lang']==l].iloc[:,0:-1]

        self.ingress_sets = {}
        for l in self.langs:
            self.ingress_sets[l] = self.serving[self.serving['wiki_lang'] == l].iloc[:,0:]

        self.lang2dc = {}
        for l in self.langs:
            self.lang2dc[l] = pd.DataFrame(self.ingress_sets[l]['ingress'].value_counts())
            self.lang2dc[l].columns=[l]
        
        self.lang2dc_mat = pd.concat([self.lang2dc['en'], \
                                     self.lang2dc['ja'], \
                                     self.lang2dc['de'], \
                                     self.lang2dc['fr'], \
                                     self.lang2dc['zh'], \
                                     self.lang2dc['ru'], \
                                     self.lang2dc['es']], sort= False, axis=1).fillna(0)
        self.lang2dc_mat['total'] = self.lang2dc_mat.sum(axis=1)
        self.lang2dc_mat.loc['Total']= self.lang2dc_mat.sum()

        self.sum_mean = {}
        for l in self.lang_sets:
            self.sum_mean[l] = self.lang_sets[l].mean(axis=0)*self.lps.loc[l,'bits_per_page']

        self.sum_quantile = {}
        for l in self.lang_sets:
            self.sum_quantile[l] = self.lang_sets[l].quantile(q=self.quantile, axis=0)*self.lps.loc[l,'bits_per_page']

        if self.quantile == 'mean':
            self.sums =  self.sum_mean.copy()
        else:
            self.sums =   self.sum_quantile.copy()

        self.Sites = ['Ashburn','Carrollton','Haarlem','San Francisco','Singapore']
        
    def get_language(self, page):
        res = re.search('[a-z][a-z].wikipedia.org',page)
        if res:
            return res[0][0:2]
        return 'na'
    
    def l_dist(self, file):
        lang_dist = Counter(self.df.lang)
        plt.bar(*zip(*sorted(lang_dist.items())))
        plt.savefig(path+file)
    

    def ingress_series(self):
        self.ingress_sums = {}
        for dc in self.dcs:
            self.ingress_sums[dc] = \
            (self.lang2dc_mat.loc[dc, 'en']/self.lang2dc_mat.loc['Total', 'en']*self.sums['en']) + \
            (self.lang2dc_mat.loc[dc, 'ja']/self.lang2dc_mat.loc['Total', 'ja']*self.sums['ja']) + \
            (self.lang2dc_mat.loc[dc, 'de']/self.lang2dc_mat.loc['Total', 'de']*self.sums['de']) + \
            (self.lang2dc_mat.loc[dc, 'fr']/self.lang2dc_mat.loc['Total', 'fr']*self.sums['fr']) + \
            (self.lang2dc_mat.loc[dc, 'ru']/self.lang2dc_mat.loc['Total', 'ru']*self.sums['ru']) + \
            (self.lang2dc_mat.loc[dc, 'zh']/self.lang2dc_mat.loc['Total', 'zh']*self.sums['zh']) + \
            (self.lang2dc_mat.loc[dc, 'es']/self.lang2dc_mat.loc['Total', 'es']*self.sums['es'])
        return self.ingress_sums

    def dc_capacities(self):
        '''Capacity for a DC inclusice of all lanuages'''
        dc_max = []
        for dc in self.dcs:
            cap = [dc, np.max(self.ingress_series()[dc])]
            dc_max.append(cap)
        #print(dc_max)
        return dc_max

    def tau(self):
        self.lang_tau = {}
        for dc in self.dcs:
            for l in self.langs:
                self.lang_tau[str(dc)+' '+str(l)] = self.dc_service_dmd(dc,l)/(self.ingress_series()[dc])
        return self.lang_tau

    def tau_lang(self, lang):
        cols = pd.DataFrame(self.tau()).columns
        _dcs = []
        for col in cols:
            if str(col)[-2:]==lang:
                _dcs.append(col)
        _dcs_mat = pd.DataFrame(self.tau())[_dcs]
        return _dcs_mat

    def tau_dc(self, dc):
        cols = pd.DataFrame(self.tau()).columns
        _dcs = []
        for col in cols:
            if str(col)[:-3]==dc:
                _dcs.append(col)
        _dcs_mat = pd.DataFrame(self.tau())[_dcs]
        _dcs_mat['total'] = _dcs_mat.sum(axis=1)
        return _dcs_mat
    
    def ingress_dist(self):
        _df = pd.DataFrame(self.serving['ingress'].value_counts())
        return(_df)

    def dc_service_dmd(self, dc, lang):
        _df = self.lang2dc_mat.loc[dc, lang]/self.lang2dc_mat.loc['Total', lang]*self.sums[lang]
        return _df

    def quantile_curves(self):
        '''Plots the curve for each language.'''
        days = [r for r in range(self.sum_quantile['en'].shape[0])]

        fig = plt.figure(1,figsize=[10,10])
        plt.ylabel('Quanitile bits viewed')
        plt.xlabel('Day')
        plt.title('Pages in Different Languages')
        labels = self.labels

        for l in self.sum_quantile:
            plt.plot(days,self.sum_quantile[l],label = labels[l])

        plt.legend()
        plt.show()
        plt.savefig(path + 'quantile_use.jpg') 

    def mean_curves(self):
        '''Plots the curve for each language.'''
        days = [r for r in range(self.sum_mean['en'].shape[0])]

        fig = plt.figure(1,figsize=[10,10])
        plt.ylabel('Mean bits viewed')
        plt.xlabel('Day')
        plt.title('Pages in Different Languages')
        labels = self.labels

        for l in self.langs:
            plt.plot(days,self.sum_mean[l],label = labels[l])

        plt.legend()
        #plt.show()
        plt.savefig(path + 'average_use.jpg')
    
    def lang2dc_curves(self):
        '''Plots the curves for all lanauges serviced at a DC.'''
        days = [r for r in range(self.sum_quantile['en'].shape[0])]

        fig = plt.figure(1,figsize=[6.4,4.8])
        plt.tick_params(axis="x", labelsize=18)
        plt.tick_params(axis="y", labelsize=18)
        plt.ylabel(str(self.quantile)+' quanitile bits viewed', fontsize=18, color='dimgrey' )
        plt.xlabel('Day', fontsize=18, color='dimgrey')
        # plt.title('Sum of Lanugages to Each DC', fontsize=18)

        for dc in self.dcs:
            plt.plot(days, self.ingress_series()[dc],label = dc)

        plt.legend(fontsize = 12)
        #plt.show()
        plt.savefig(path + 'lang2dc_curve_1.jpg' , bbox_inches='tight')

    def tau_curves(self, lang):
        '''Plots the curves for all lanauges serviced at a DC. For example in SFO the work is 100% English throughout.'''
        days = [r for r in range(self.sum_quantile['en'].shape[0])]

        fig = plt.figure(1,figsize=[10,10])
        plt.ylabel('tau')
        plt.xlabel('Day')

        for dc in self.dcs:
            plt.title('tau for selected language: '+str(lang))
        
            plt.plot(days, self.get_lang_tau(lang, dc),label = dc)

        plt.legend()
        plt.show()
        #plt.savefig(path +str(lang)+ '_tau_curve.jpg')

    def information(self):
        # for l in self.langs:
        #     print('This is self.langs(): '+ l)
        # print('\n')
        # for l in self.sum_mean:
        #     print('This is for self.sum_mean(): '+ l)
        print('\nData Center Locations: \n'+str(self.dcs))
        print('\nThis is ingress_distribution across those Data Centers : \n' +str(self.ingress_dist()))
        print('\nLanguages or Services being analyzed: \n'+str(self.langs))
        print('\nThis is the distribution of languages to the set of data-centers: \n' +str(self.lang2dc_mat))
        print('\nThis is self.sum_mean["en"]: \n'+ str(self.sum_mean['en']))
        print('\nCountry/Market to lanuage/service mapping: \n' +str(self.serving))
        # print('\n')
        # for l in self.ingress_sets:
        #     print('This is for self.ingress_sets(): '+ l)
        # print('\n')
        # print('Example of country/market to data-center mapping for english:\n' +str(self.ingress_sets['en'].head()))
        print(colored('\nThis is the time series demands at each data-center. SIN @ 95%tile: \n'+str(self.ingress_series()['Singapore']), 'red'))
        print(colored('\nCapacity of each data-center, inclusive of all lanuages:\n'+str(self.dc_capacities()), 'magenta'))
        print(colored('\nA single data-center\'s time series. Here is demand for English dialects in SIN: \n'+str(self.dc_service_dmd('Singapore', 'en')),'blue'))
        print('\nThis is the ingress series.\n', str(pd.DataFrame(self.ingress_series())))
        print('\ntau, the service load share.\n' +str(pd.DataFrame(self.tau())))
        print('\ntau, the service load share for English dialects in Singapore.\n' +str(pd.DataFrame(self.tau()['Singapore en'])))
        print(colored('\nThis shows the relative demand of each language to each data-center. Here is \'es\': \n'+str(self.tau_lang('es')), 'blue'))
        print(colored('\nThis shows the relative demand at each data-center for each language. Here is \'Ashburn\': \n'+str(self.tau_dc('Ashburn')),'green'))

    def sys_cost_D(self, lang, p = None):
        if p != None:
            print(colored('Derive array D for Algorimthm 1.', 'green'))
            print(colored('D:', 'magenta'))
        
        def D(lang):
            _D = self.tau_lang(lang)
            return _D
        
        _df = D(lang)

        if p != None:
            print(colored(_df, 'magenta'))
        return _df

    def get_lang_tau(self, lang, dc):
        '''Results in curves that indicate the workload each service is demanding from each site. '''
        _D = self.sys_cost_D(lang)
        _D = _D[[str(dc)+' '+str(lang)]]
        #print(colored('\ntau for ' + lang +' at '+dc+':\n '+str(_D), 'green'))
        #print(colored(str(_D)))

        return _D

    def get_cap_lang_dc(self, lang):
        '''This is the max workload that the facility must be planned for.'''
        cap_dc = []
        for dc in self.dcs:
            _cap = [dc, np.max(self.get_lang_tau(lang, dc).values)]
            cap_dc.append(_cap)
        print(cap_dc)
        return cap_dc
    
    def get_traffic(self, lang):
        tau_dict = {}

        for site in self.Sites:
            tau_dict[site] = self.tau()[str(site) + ' ' + lang][-365:]

        return tau_dict

    def expand_traffic(self, lang):
        '''Modulo expansion form daily to hourly traffic'''
        dmd = {}
        for site in self.Sites:
            traffic_list = self.get_traffic(lang)[site]
            init_val = traffic_list[0]
            dmd[site]=[init_val]
            for i in range(1,8760):
                dmd[site].append(traffic_list[int(i/24)])
        return dmd



def main():
    test = traffic(train_1, 0.50)
    # test.lang2dc_curves()
    print(test.information())
    #test.sys_cost_D('en',p = True)
    # test.tau_curves('en')
    # print(test.get_lang_tau('zh','San Francisco').sum())
    #test.get_cap_lang_dc('zh')
    # print(test.expand_traffic('en')['Ashburn'])
    
    

if __name__ == "__main__":
  main()