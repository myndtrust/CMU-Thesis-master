import gym
import pandas as pd
import numpy as np
from termcolor import colored
import streamlit as st
#import tensorflow as tf
import random
import sys
import os
sys.path.insert(0, "/home/eric/ramukcire/estimating_cost_of_dc_services/syscost/")
sys.path.insert(1, "/home/eric/ramukcire/estimating_cost_of_dc_services/syscost/bem/Gym_Eplus_master/")
sys.path.insert(1, "/home/eric/ramukcire/estimating_cost_of_dc_services/3rdparty/energysimulation/")
import traffic.traffic as traffic
import dyna_mecbem as dc

import gridsim.grid_sim_linear_program as gslp
import gridsim.grid_sim_simple_example as gsse
import os.path as osp
import matplotlib
import matplotlib.pyplot as plt
#print(colored((sys.path), 'red'))


class dc_mec(object):
    def __init__(self, lang, rps=None, traffic_file='unit_testing.csv', quantile=0.50):
        self.lang = lang
        self.rps = rps
        self.traffic = traffic_file
        self.quantile = quantile
        self.Sites = traffic.traffic(self.traffic, self.quantile).Sites
        self.Countries = ['United States','United States','Netherlands','United States', 'Singapore' ]
        self.Regions = ['Midatlantic', 'ERCOT', 'Netherlands', 'California', 'Singapore' ]
        self.image_dir = '/home/eric/ramukcire/estimating_cost_of_dc_services/Manuscript/images/'
        self.df_path = '/home/eric/ramukcire/estimating_cost_of_dc_services/syscost/mec/'

        self.NG = gslp.GridSource(
            name='NG',  # Dispatchable, so no name restriction.                                                                                                
            nameplate_unit_cost=1239031,  # Cost for a combined cycle plant. $/MW                                                                              
            variable_unit_cost=17.5,  # Cheap fuel costs assumes fracking. $/MWh                                                                               
            co2_per_electrical_energy=0.33,  # Tonnes CO2 / MWh                                                                                                
            is_rps_source=False)  # Not in Renewable Portfolio Standard. 
 
        self.solar = gslp.GridSource(
            name='SOLAR',  # Matches profile column name for nondispatch.                                                                                      
            # nameplate_unit_cost=946000,  # Aggressive solar cost $/MW 
            nameplate_unit_cost = 1.5e6, # $ / Mw  # Cost more indicative to current market.                                                                                        
            variable_unit_cost=0,  # No fuel cost.                                                                                                             
            co2_per_electrical_energy=0,  # Clean energy.                                                                                                      
            is_rps_source=True)  # In Renewable Portfolio Standard 

        self.wind = gslp.GridSource(
            name='WIND',  # Matches profile column name for nondispatch.                                                                                      
            nameplate_unit_cost=2051532,  # Aggressive wind cost $/MW                                                                                          
            variable_unit_cost=0,  # No fuel cost.                                                                                                             
            co2_per_electrical_energy=0,  # Clean energy.                                                                                                      
            is_rps_source=True)  # In Renewable Portfolio Standard

        self.Coal = gslp.GridSource(name='COAL')


    def dc2region(self):
        d = dict(zip(self.Sites, self.Regions))
        return d


    def get_dmd(self):
        df= traffic.traffic(self.traffic, self.quantile)
        dmd = df.expand_traffic(self.lang)
        return dmd


    def get_bem_all(self):
        dmd = self.get_dmd()
        agent = {}
        dc_pwr = {}

        for site in self.Sites:
            init_dc = site.replace(' ','')
            agent[site] = dc.agent('Eplus-dc_'+init_dc+'-v1', self.lang)
            dc_pwr[site] = agent[site].schedule_run(site, dmd=dmd)
            
        return dc_pwr
        

    def get_pwr_site(self, site, df_only=False):
        dmd = self.get_dmd()
        agent = {}
        dc_pwr = {}

        init_dc = site.replace(' ','')

        agent[site] = dc.agent('Eplus-dc_'+init_dc+'-v1', self.lang, df_only, df_path=self.df_path)
        dc_pwr= agent[site].schedule_run(site, dmd=dmd)
            
        return dc_pwr['Total-Facility_Power[w]']

    def plot_pwr_site(self):
        for site in self.Sites:
            y = self.get_pwr_site(site)
            y.plot(label=site)
        plt.legend()
        plt.show()

    def get_bem_df(self):
        for site in self.Sites:
            self.get_pwr_site(site, df_only=True)

        

    def get_renews(self):
        df = self.get_csv('data/world_energy_gens.csv')
        df = df[df['Country'].isin(self.Countries)]
        df.columns = [s.replace('\n', ' ') for s in df.columns.tolist()]
        df.columns = ['Country', 'Year', 'f_Renewables',\
                      'Total_Generation (GWh)', 'Total renewable (GWh)',\
                      'Hydropower (GWh)', 'Wind (GWh)', 'Biomass and waste (GWh)',\
                      'Solar(GWh)', 'Geothermal (GWh)', 'Wave and tidal (GWh)']
        return df


    def get_regions(self):
        regions = os.listdir("/home/eric/ramukcire/estimating_cost_of_dc_services/3rdparty/energysimulation/gridsim/data/profiles/") 
        return regions
    

    def print_regions(self):
        regions = self.get_regions() 
        regions = list(enumerate(regions))
        return regions
    

    def get_mecprofile(self, region, site=None, with_dc=False, prnt=False): 
        if prnt:
            print(region)
            print(self.print_regions())
        profiles_path = gsse.get_data_directory() + ['profiles', region]
        profiles_file = osp.join(*profiles_path)
        profiles = pd.read_csv(profiles_file, index_col=0, parse_dates=True)
        if with_dc:
            profiles.DEMAND = profiles.DEMAND + self.get_pwr_site(site)/1000 # Scale to 1000 accounts for W to MW conversion. 1e6W/MW. 
                                                                            # The power density for hyperscale DC is 10x-20x in size. 1e5W/MW.
                                                                            # Power density is also 10x-20x 1e4W/MW
                                                                            # Consider a 300MW Site 1e3W/MW
        annual_average = profiles.DEMAND.sum()/8760
        
        for col in profiles.columns.tolist():
            profiles[col]=profiles[col].astype(float)

        return annual_average, profiles
        


    def build_lp(self,region, site=None, with_dc=False, NG=True, solar=True, wind=True):
        # _, profiles = self.get_mecprofile(region, site, with_dc)
        profiles = self.get_mecprofile(region, site, with_dc)[1]
        lp = gslp.LinearProgramContainer(profiles)
        lp.add_demands(gslp.GridDemand('DEMAND'))
        #lp.add_dispatchable_sources(self.Coal)

        if NG:
            lp.add_dispatchable_sources(self.NG, self.Coal)
        if solar:
            lp.add_nondispatchable_sources(self.solar)  
        if wind:
            lp.add_nondispatchable_sources(self.wind)

        
        for source in lp.sources:
            print(source.name)


        # Also adjust lp based upon what we think future costs are going to be.
        hours_per_year = 24 * 365
        annual_discount_rate = 0.06
        lifetime_in_years = 30

        lp.carbon_tax = 0 # No Carbon Tax
        lp.rps_percent = self.rps # No renewable requirement
        lp.cost_of_money = gslp.extrapolate_cost(
            1.0,
            annual_discount_rate,
            lp.number_of_timeslices / hours_per_year,
            lifetime_in_years)
        
        lp.solve()

        return lp
    


    def source_stats(self, region):
        lp =self.build_lp(region)
        index = []
        capital = []
        variable = []
        co2_per_mwh = []
        nameplate = []
        generated = []
        
        for s in lp.sources:
            index.append(s.name)
            capital.append(s.nameplate_unit_cost)
            variable.append(s.variable_unit_cost)
            co2_per_mwh.append(s.co2_per_electrical_energy)
            nameplate.append(s.get_nameplate_solution_value())
            generated.append(sum(s.get_solution_values()))
            
        data = {'capital': capital,
            'variable': variable,
            'co2_per_mwh': co2_per_mwh,
            'nameplate': nameplate,
            'generated': generated}
        
        df = pd.DataFrame(index=index,
                        data=data)
        df['cost_of_money'] = lp.cost_of_money
        df['carbon_tax'] = lp.carbon_tax
        df['total_cost'] = (df.capital * df.nameplate +
                            df.cost_of_money * df.generated *
                            (df.variable + df.co2_per_mwh * df.carbon_tax))
        return df


    def plot_labels(self, region):
        lp =self.build_lp(region)
        plt.xlabel('Hour of Year')
        plt.ylabel('MegaWatts')
        
        title_string = region[9:-4].upper()+' Dmd and Srcs: \n'
        title_string += 'RPS = %d, Carbon Tax = %d' %(lp.rps_percent,
                                                    lp.carbon_tax)
        plt.title(title_string)
        plt.legend(bbox_to_anchor=(1.25, 1.0))  


    def plot_sources(self, region, site=None, with_dc=False, slicee=None):
        linestyles = ['dashed', 'dotted', '-.', ':','solid']
        colors = ['red', 'aqua', 'chartreuse', 'coral', 'fuchsia', 'magenta','black']
        lp = self.build_lp(region, site, with_dc)
        if slicee is None:
            slicee = slice(lp.number_of_timeslices)
        profiles = lp.profiles
        plt.plot(profiles.index[slicee], 
                profiles.DEMAND[slicee], label='Demand', linestyle=linestyles[0], color = colors[0])
        for i in range(len(lp.sources)):
            plt.plot(profiles.index[slicee],
                    lp.sources[i].get_solution_values()[slicee], label=lp.sources[i].name, linestyle=linestyles[i+1], color=colors[1+i])
        self.plot_labels(region)
        # #plt.savefig(self.image_dir+'lp_'+str(region)[9:-4]+'_'+str(site)+'.png', bbox_inches='tight')
        plt.show()
        plt.close()
        # print(profiles)

    def get_csv(self, path):
        df = pd.read_csv(path, header=0)
        return df


def main():
    dc_mc = dc_mec('en',rps=30)
    region = dc_mc.get_regions()[13]
    print(region)
    site = 'San Francisco'
    # print(dc_mc.dc2region())
    # print(dc_mc.Sites)
    print(dc_mc.get_dmd()['San Francisco'])
    # print(dc_mc.get_bem_all())
    # print(dc_mc.get_pwr_site(site))
    #print(dc_mc.plot_pwr_site())
    # dc_mc.get_bem_df()
    # print(dc_mc.print_regions())
    # print(dc_mc.get_renews())
    # print(dc_mc.get_regions())
    # print(dc_mc.get_regions()[0])
    # print(dc_mc.get_mecprofile(region)[1])
    # print('Annual Region Only Average DEMAND: ', dc_mc.get_mecprofile(region))
    # print((dc_mc.get_mecprofile(dc_mc.get_regions()[13], site, with_dc=False)[1]).info())
    # print((dc_mc.get_mecprofile(dc_mc.get_regions()[7], site, with_dc=False)[1]).info())
    # print('Annual with_dc Average DEMAND: \n', dc_mc.get_mecprofile(region, site, with_dc=False)[1])
    # print(dc_mc.build_lp(dc_mc.get_regions()[1]))
    # print(dc_mc.source_stats(region))
    # dc_mc.plot_sources(region, site)
    # dc_mc.plot_sources(region, site, with_dc=False)
    # dc_mc.plot_sources(region, slicee=slice(4000,4200))
    # dc_mc.build_lp(region, site=site, with_dc=False, NG=True, solar=True, wind=True)

    #print(dc_mc.get_mecprofile(region, site, with_dc=True))
    # print(dc_mc.get_renews().columns)
    #print(dc_mc.source_stats(region))

if __name__ == "__main__":
  main()
