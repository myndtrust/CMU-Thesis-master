    #!/usr/bin/python3
import sys
sys.path.insert(0, "/home/eric/ramukcire/estimating_cost_of_dc_services/syscost/")
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from collections import Counter
import itertools
import warnings
from termcolor import colored
import streamlit as st
from subprocess import check_output
import traffic.traffic as traffic
# from traffic.traffic import traffic 
from datetime import datetime as dt

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

print(colored('Imported Modules\n', 'yellow'))
print(colored('Running from '+str((os.getcwd())),'green'))
#print(colored('Other directories at this level are '+ str(os.listdir()),'red'))

st.title('Total Cost of Ownership Model (Hardy)')
st.subheader('Eric Kumar PE, Doctor of Design')

'''This script will run the Hardy model. For now it will not interact \
    with the model directly, but will be able to consume the outputs \
    from the Perl program, parse it, and pass it to SysCost EEIO inputs. \
        '''

class costet(object):
    def __init__(self, input_dc, input_r, streamlit=True, model_file=None):
        '''Args: Runs the specified parameter TCO model.
                 input_dc: "input_example/dc.params" (Data-Center Parameters)
                 input_r: "input_example/r.params" (Resource Parameters)
                 streamlit = boolean for using streamlit
                 model_file = file name for the model output'''
        self.input_dc = input_dc
        self.input_r = input_r
        self.model = check_output(["perl", "./cost-et.pl", input_dc, input_r], shell = False)
        self.model = self.model.decode("utf-8")
        self.streamlit = streamlit
        self.model_file = model_file

    def view_raw_output(self, save=None):
        if self.streamlit is True:
            st.header('Model run for ' +self.input_dc+' with '+self.input_r)
            st.subheader('Output from Cost-ET Model run')
            st.text(self.model)
        if save is not None:
            f = open(self.model_file, "w+")
            f.write(str(self.model))
            f.close()
        print(colored('This is the output from the Cost-ET model: ' + self.model, 'yellow'))

    def view_script(self, script):
        '''Args: script: "cost-et.pl" '''
        f = open(script, "r")
        f = f.read()
        print(colored('Print this :'+ f, 'magenta'))
        if self.streamlit is True:
            st.subheader('Print out of '+script)
            st.code(f, language='perl')

    def get_dc_params(self):
        _df = pd.read_csv(self.model_file)[2:24].reset_index(drop=True)
        _df.columns = ['DC_parameters']
        _df[['DC Param','Value']] = _df['DC_parameters'].str.split("=",expand=True)
        _df = _df[['DC Param','Value']]
        if self.streamlit is True:
            st.subheader('DC Parameters: ')
            st.dataframe(_df, 500, 600)
        return _df

    def get_resource_params(self):
        _df = pd.read_csv(self.model_file)[29:76].reset_index(drop=True)
        _df.columns = ['Resource_parameters']
        _df[['Resource','Value']] = _df['Resource_parameters'].str.split("=",expand=True)
        _df = _df[['Resource','Value']]
        if self.streamlit is True:
            st.subheader('Resources Parameters: ')
            st.dataframe(_df, 500, 600)
        return _df

    def get_server_age(self):
        _df = pd.read_csv(self.model_file)[79:85].reset_index(drop=True)
        _df.columns = ['Age Dist']
        _df[['Age (Years)','Server Count']] = _df['Age Dist'].str.split(" ",expand=True)
        _df = _df[['Age (Years)','Server Count']]
        if self.streamlit is True:
            st.subheader('Age: ')
            st.dataframe(_df, 500, 1000)
        return _df

    def get_server_replacement(self):
        '''Unclear what this calue means ATM.'''
        _df = pd.read_csv(self.model_file)[85:86].reset_index(drop=True)
        _df.columns = ['Server Replacements']
        _df[['Count','Server Count']] = _df['Server Replacements'].str.split(" ",expand=True)
        _df = _df[['Count']]
        if self.streamlit is True:
            st.subheader('Server Replacement: ')
            st.dataframe(_df, 500, 1000)
        return _df

    def get_dc_costs(self):
        # This requires that the model be run. To create a new model txt file.
        _df = pd.read_csv(self.model_file)[90:96].reset_index(drop=True)
        _df.columns = ['DC Costs']
        _df[['Cost_Component', 'Cost', 'Unit', 'Relative']] = _df['DC Costs'].str.split(" ",expand=True)
        _df = _df[['Cost_Component','Cost','Unit', 'Relative']].iloc[1:,:]
        _df['Cost_Component'] = _df['Cost_Component'].str[:-1]
        _df.set_index('Cost_Component', inplace = True)
        #_df.index = _df['Cost_Component']
        if self.streamlit is True:
            st.subheader('Data Center Costs: ')
            st.dataframe(_df, 500, 1000)
        return _df

    def get_dc_tco(self):
        _df = pd.read_csv(self.model_file)[90:96].reset_index(drop=True)
        _df.columns = ['DC Costs']
        _df[['Cost_Component','Cost','Unit', 'Relative']] = _df['DC Costs'].str.split(" ",expand=True)
        _df = _df[['Cost_Component','Cost','Unit']].iloc[:1,:]
        _df['Cost_Component'] = _df['Cost_Component'].str[:-1]
        # _df.set_index('Cost_Component', inplace = True)
        if self.streamlit is True:
            st.subheader('Data Center Total Cost of Ownership: ')
            st.dataframe(_df, 500, 1000)
        return _df
    
    def plot_dc_costs(self, plot):
        '''Create and Save figure. The figure is then loaded as an image into Streamlit.
           Args: plot: the file to save name to save the figure as'''
        _df = self.get_dc_costs()
        plot_file = '../images/'+str(plot)+'.png'
        _plt = _df[['Cost']].copy()
        _plt.Cost = _plt.Cost.astype(float)
        _plt.plot(kind='bar')
        plt.savefig(plot_file)
        if self.streamlit is True:
            st.subheader('Plot of Data Center Costs: ')
            st.image(plot_file)
        print(_plt)

    def res_dims(self):
        '''Provides the units of input and outputs'''
        _df = pd.read_csv(self.model_file)[96:102]
        _df.columns = ['Resource Dims']
        _df = _df['Resource Dims'].str.split(":",expand=True)
        _df[1].str.lstrip(' ')
        _df[1] = _df[1].replace('\t',' ', regex=True) 
        regex = _df[1].str.extract(r'(\d+\.*\d*)\s*(\w+\^*\d*)*', expand = True)
        _df = pd.concat([_df, regex], axis =1)
        _df.columns = ['parameter','mix_value', 'value','units']
        _df.set_index('parameter', inplace = True)
        _df = _df[['value', 'units']].fillna(' ')
        if self.streamlit is True:
            st.subheader('Resource Dimensions: ')
            st.dataframe(_df, 500, 1000)

class eeio(object):
    def __init__(self, input_csv):
        self.Introduction = pd.read_pickle('eeio_data/Introduction.pkl')
        self.Environmental_inventory = pd.read_pickle('eeio_data/Environmental_inventory.pkl')
        self.TRACIimpacts_of_selected_effect = pd.read_pickle('eeio_data/TRACIimpacts of selected effect.pkl')
        self.Select_activity_in_a_sector = pd.read_pickle('eeio_data/Select_activity_in_a_sector.pkl')
        self.Leontief_inverse_of_A = pd.read_pickle('eeio_data/Leontief_inverse_of_A.pkl') #total requirements table (or matrix)
        self.TRACI_impacts = pd.read_pickle('eeio_data/TRACI_impacts.pkl')
        self.Economic_impact = pd.read_pickle('eeio_data/Economic_impact.pkl')
        self.Economic_impact_ranked = pd.read_pickle('eeio_data/Economic_impact_ranked.pkl')
        self.Emission_employmeDiscussionnt_impacts = pd.read_pickle('eeio_data/Emission_employment_impacts.pkl')
        self.Environment_intensity_matrix = pd.read_pickle('eeio_data/Environment_intensity_matrix.pkl')
        self.Transpose_of_Env_matrix = pd.read_pickle('eeio_data/Transpose_of_Env_matrix.pkl')
        self.TRACI_characterization_factor = pd.read_pickle('eeio_data/TRACI_characterization_factor.pkl')
        self.Transpose_of_Env_matrix = pd.read_pickle('eeio_data/Transpose_of_Env_matrix.pkl')
        self.Transpose_of_TRACI_factor = pd.read_pickle('eeio_data/Transpose_of_TRACI_factor.pkl')
        self.commodities = pd.read_csv('sectors.csv').Sector.to_list()
        self.input_df = pd.read_csv(input_csv) 

    def get_sectors(self):
        df = self.input_df[['Sector']]
        return df

    def get_costs(self):
        df = self.input_df[['Relative Costs']]
        return df
    
    def Matrix_A(self):
        df = pd.read_pickle('eeio_data/Matrix_A.pkl') #direct  requirements  table(or  matrix).
        df.rename(index = {'2122a0/iron, gold, silver, and other metal ores/us ': \
                           '2122a0/iron, gold, silver, and other metal ores/us'}, inplace = True)
        return df

    def I_matrix(self):
        df = pd.read_pickle('eeio_data/I_matrix.pkl')
        df = df.astype(float)
        return df

    def IplusA(self):
        df = self.I_matrix() + self.Matrix_A()
        return df

    def IplusA_total(self):
        df = self.IplusA().sum(axis=(1))
        return df

    def IminusA(self):
        df = np.subtract(self.I_matrix(), self.Matrix_A())
        return df

    def IminusA_total(self):
        df = self.IminusA().sum(axis=(1))
        return df

    def Inv_IminusA(self):
        df = pd.DataFrame(np.linalg.inv(self.IminusA()))
        df.index = self.IminusA().index
        df.columns = self.IminusA().columns
        return df

    def activity(self):
        act = self.Select_activity_in_a_sector.set_index('Industry code/Industry name', drop=True)
        return act #[act['Amount of economic activity(2013 dollar)']==0]

    def search(self, term):
        s = []
        for word in term:
            activity = self.activity()
            w=activity[activity.index.str.contains(word)].index
            s.append(w)
        return pd.DataFrame(s, index=[term]).T#.fillna('-')

    def input(self):
        sectors = self.get_sectors()
        costs = self.get_costs()
        activity = self.activity()
        for i in range(len(sectors)):
            activity.loc[sectors.iloc[i], 'Amount of economic activity(2013 dollar)'] = float(costs.iloc[i])
            inputs = activity[activity['Amount of economic activity(2013 dollar)']!=0]
        return activity, inputs

    def direct_costs(self):
        df = self.IplusA()@self.input()[0]
        df.columns=['direct_costs']
        return df

    def total_costs(self):
        df = self.Inv_IminusA()@self.input()[0]
        df.columns = ['total_costs']
        return df

    def econ_impacts(self, rank='total'):
        df = pd.concat([self.total_costs(),self.direct_costs()], axis=1)
        if rank == 'total':
            df = df.sort_values(by=['total_costs'], ascending = False)
        elif rank == 'direct':
            df = df.sort_values(by=['direct_costs'], ascending = False)
        return df

    def level_sectors(self, cost='total'):
        if cost == 'total':
            df = self.total_costs()
        elif cost == 'direct':
            df = self.direct_costs()
        df = df.loc[df.index.isin(self.commodities)]
        return df

    def env_inventory(self):
        df = pd.DataFrame(self.Environment_intensity_matrix@self.Inv_IminusA())
        total_env_vector = df.sum(axis=1)
        return df, total_env_vector

    def emission_emp_impacts(self):
        env_inv_t = self.Environment_intensity_matrix.T.values
        df = pd.DataFrame(env_inv_t * self.econ_impacts(rank='total')[['total_costs']].values)
        df.columns = self.Environment_intensity_matrix.T.columns
        df.index= self.econ_impacts(rank='total')[['total_costs']].index 
        total_emissions_impact = df.sum()
        return df, total_emissions_impact

    def TRACI(self):
        df = self.TRACI_characterization_factor@self.env_inventory()[0]
        return df

    def TRACI_impacts_of_selected_effect(self):
        df = self.emission_emp_impacts()[0]@self.TRACI_characterization_factor.T
        return df

class buildings(object):
    """Args:
        power: the nominal power rating of the data-center in kW.
        over-sub: over subcription ratio"""
    def __init__(self, power, osr=0):
        self.power = power # power in KW
        self.osr = osr

    def get_building(self, column = "Relative Costs", out_csv = False ):
        Y = pd.read_csv('Y_building.csv')
        Y = Y[['Sector', column]]
        Y[column] = Y[column]*self.power*1000 # Mulitpling by a thosand to convert from $/W to $/kW
        if out_csv == True: 
            Y.to_csv('Y_building_'+ str(self.power) + '.csv')
        return Y

class servers(object):
    """Args:
        power: the nominal power rating of the data-center in kW.
        over-sub: over subcription ratio"""
    def __init__(self, power, osr=0):
        self.power = power # power in KW
        self.osr = osr
    
    def read_comp(self, df='component_power.csv'):
        d = pd.read_csv(df).fillna(0)
        d = d[['Component','Units', 'Power Budget', 'Mix-SSD','Mix_HDD_only', \
            'High IO', 'Compute', 'Mix_low_memory', 'Units.2']]
        return d

    def server_power(self, summed=False):
        cpt_pwr = self.read_comp().set_index('Component')
        power = {}
        for i in ['Mix-SSD', 'Mix_HDD_only', 'High IO', 'Compute', 'Mix_low_memory']:
            power[i] = cpt_pwr['Power Budget'] * cpt_pwr[i]
        df = pd.DataFrame(data=power)
        if summed != False: 
            df.loc["sum"] = df.sum(axis=0)
        return df

    def get_servercosts(self):
        df = pd.read_csv('server_costs.csv').T
        df.columns = ['costs']
        return df

    def server_budget(self, type_server='Mix-SSD', network=0.05):
        MW = (self.power * 1000)*(1-network)
        type_cost = self.get_servercosts()['costs'].loc[type_server]
        type_power = self.server_power(summed=True)[type_server].loc['sum']
        count_server = MW/type_power
        cost_server = count_server*type_cost
        return count_server, cost_server

    def get_server_Y(self, type_server='Mix-SSD', network=0.05, out_csv=False):
        budget = self.server_budget(type_server=type_server, network=network)[1]
        SKU_distribution = pd.read_csv('SKU_Parts_share.csv').set_index('EEIO Sector' ,drop=True)[type_server]
        Y = pd.DataFrame(budget * SKU_distribution)
        Y.columns= ['Relative Costs']
        nw_cost = budget * network
        nw_sector = '334290/communications equipment/us' 
        data = {nw_sector:[nw_cost]} 
        nw_Y = pd.DataFrame(data, index=['Relative Costs'] ).T
        Y = Y.append(nw_Y)
        Y['Sector'] =  Y.index
        if out_csv == True:
            Y.to_csv(type_server+'_'+str(self.power)+'.csv')
        return Y

class syscost(object):
    """Args:
        power: the nominal power rating of the data-center in kW.
        out_csv: write the output csv"""
    def __init__(self, power, out_csv = False):
        self.power = power
        self.out_csv = out_csv

    def total_embodied(self, cns_life=3,dc_life=20, type_server='Mix-SSD' ):
        # building = buildings(power)
        server = servers(self.power)
        server.get_server_Y(type_server, out_csv=self.out_csv)
        server_csv = type_server+'_'+str(self.power)+'.csv'
        cns = eeio(server_csv)
        em_cns = cns.emission_emp_impacts()[1]/cns_life
        traci_cns = cns.TRACI_impacts_of_selected_effect().sum(axis=0)

        building = buildings(self.power)
        building.get_building(out_csv=self.out_csv)
        building_csv = 'Y_building_'+ str(self.power) + '.csv'
        dc = eeio(building_csv)
        em_dc = dc.emission_emp_impacts()[1]/dc_life
        traci_dc = dc.TRACI_impacts_of_selected_effect().sum(axis=0)

        total_emissions = em_cns + em_dc
        total_traci = traci_dc + traci_dc

        return pd.DataFrame(total_emissions), pd.DataFrame(total_traci)

    def energy(self):
        df = self.total_embodied()[0]
        df = df[df.index.str.slice(start=-2)=='mj']
        return df

class dc_traffic(object):
    '''
    Args: Power (kW)
    '''
    def __init__(self, power, lang='en', traffic_file='train_2.csv', quantile=0.50):
        self.lang = lang
        if traffic_file != 'train':
            self.traffic = 'train_2.csv'
        else:
            self.traffic = 'unit_testing.csv'
        self.quantile = quantile
        self.power = power
        
    def embodied(self, denominator=8760):
        '''This is the total hourly embodied costs of the DC, inclusive of all services.
        Args:
            denominator: defaults to 8,760 (hours). 1 = full year
        '''
        emb_cls = syscost(self.power, out_csv = False)
        emb_costs = emb_cls.total_embodied(cns_life=3, dc_life=20, type_server='Mix-SSD' )
        return emb_costs[0]/denominator, emb_costs[1]/denominator


    # def get_dmd(self):
    #     df= traffic.traffic(self.traffic, self.quantile)
    #     dmd = df.expand_traffic(self.lang)
    #     return dmd

    # def dc_srv_costs(self, site='San Francisco'):
    #     cost = self.embodied()[1]
    #     cost_ar = np.array(cost[0])

    #     hour = np.array(self.get_dmd()[site])

    #     df = []
    #     for i in cost_ar:
    #         df.append((i * hour))
    #     df = pd.DataFrame(df,index=list(cost.index))

    #     return df.T.sum()

    def langs(self):
        '''This function out a dict{Site : hour_coeifficient}
        '''
        traffic_cls= traffic.traffic(self.traffic, self.quantile)
        # dmds = traffic_cls.expand_traffic(self.lang)
        langs = traffic_cls.langs
        return langs
    
    def langs_dmd(self):
        '''This function out a dict{Site : hour_coeifficient}
        '''
        traffic_cls= traffic.traffic(self.traffic, self.quantile)
        dmds = traffic_cls.expand_traffic(self.lang)
        return dmds
    

    def global_env_cost(self):
        '''This function out a dict{Site : hour_coeifficient}
        '''
        traffic_cls= traffic.traffic(self.traffic, self.quantile)
        sites = traffic_cls.Sites
        dmds = traffic_cls.expand_traffic(self.lang)
        cost = self.embodied()[1]
        cost_ar = np.array(cost[0])
    
        srv_cost_dict = {}
        for site in sites:
            hour = np.array(dmds[site])
            df = []
            if hour.astype(float).sum !=0:
                for i in cost_ar:
                    df.append((i * hour))
            else:
                for i in cost_ar:
                    df.append(i*np.zeros(len(hour)))

            df = pd.DataFrame(df,index=list(cost.index))
            srv_cost_dict[site] = df.T.sum()

        return pd.DataFrame(srv_cost_dict)
    
    # def global_env_cost(self):
    #     traffic_cls= traffic.traffic(self.traffic, self.quantile)
    #     sites = traffic_cls.Sites

    #     global_env = {} 
    #     for site in sites:
    #         global_env[site] = self.dc_srv_costs(site=site)

    #     return pd.DataFrame(global_env)

def langs_dmd(power):
    langs_dmd = {}
    langs = ['en','ja','de','fr','zh','ru','es']

    for lang in langs:
        traffic = dc_traffic(power = power, lang=lang, traffic_file='train')
        df = traffic.langs_dmd()
        langs_dmd[lang] = df
    return langs_dmd

def lang_env_costs(power, langs):
    env_costs = {}
    # langs = ['en','ja','de','fr','zh','ru','es']
    # langs = ['en']

    for lang in langs:
        traffic = dc_traffic(power = power, lang=lang, traffic_file='train')
        df = traffic.global_env_cost()
        env_costs[lang] = df
        print("Finished: "+lang)
    return env_costs


def lang_env_csv(power, path=None):
    
    # langs=['en','ja','de','fr','zh','ru','es']
    langs=['es']
    for lang in langs:
        df = []
        file = lang+'.csv'
        traffic = dc_traffic(power = power, lang=lang, traffic_file='train')
        df = traffic.global_env_cost().T
        df = df[['impact potential/gcc/kg co2 eq']]
        df['impact potential/gcc/kg co2 eq'] = df['impact potential/gcc/kg co2 eq']*1000
        df.columns = [lang]
        df.to_csv(path+file)


    # langs = ['ja','de','fr','zh','ru','es']
    # langs = ['ja']
    # for lang in langs:
    #     file = lang+'.csv'
    #     traffic = dc_traffic(power = power, lang=lang, traffic_file='train')
    #     df_1 = traffic.global_env_cost().T
    #     df_1 = df_1[['impact potential/gcc/kg co2 eq']]
    #     df_1['impact potential/gcc/kg co2 eq'] = df_1['impact potential/gcc/kg co2 eq']*1000
    #     df_1.columns = [lang]
    #     df[lang] = df_1[lang]
    # return df




def get_matrix(costs='direct'):
    """Table used to compare the most appropriate building class for DCs"""
    health_care = eeio(['233210/health care buildings/us'], [1])
    manu_bldg = eeio(['233230/manufacturing buildings/us'], [1])
    util_bldg = eeio(['233240/utilities buildings and infrastructure/us'], [1])
    _df = util_bldg.level_sectors()
    _df.columns = ['util_bldg']
    if costs == 'direct':
        _df['manu_bldf'], _df['health_care']=manu_bldg.level_sectors('direct')['direct_costs'], health_care.level_sectors('direct')['direct_costs']
    elif costs == 'total':
        _df['manu_bldf'], _df['health_care']=manu_bldg.level_sectors()['total_costs'], health_care.level_sectors()['total_costs']
    return _df


def main():
    # print(dt.now())['en']
    # print(langs_dmd(550*1000))
    traffic = dc_traffic(power = 550, lang='en', traffic_file='train')
    # traffic.global_env_cost().T.to_csv('/home/eric/ramukcire/CMU-Thesis-master/embodied_cost_model/contents/eng_env_cost.csv')
    lang_env_csv(550,'/home/eric/ramukcire/CMU-Thesis-master/embodied_cost_model/contents/data/')
    # lang_env_csv(550, '/home/eric/ramukcire/CMU-Thesis-master/embodied_cost_model/contents/data/')
    # # print(dt.now())['en']
    # print(traffic.global_env_cost())
    # # print(traffic.langs())
    # print(dt.now())
    # print(lang_env_costs(550*1000))
    # print(dt.now())
    # print(traffic.helper())
    # print(traffic.dc_srv_costs())
    # print(df.index )
    # df['Index'] = traffic.embodied()[1].index()
    # for i in traffic.dc_srv_costs():
    #     print(i)
    # print((traffic.embodied(denominator=1)[1])) # embodied costs per hour
    # print(type(traffic.get_dmd()['San Francisco'])) # fraction of dc supporting particular language.
    # print(traffic.get_dmd()['San Francisco']*np.random.rand(8760,1))
    # for p in traffic.get_dmd():
    #     print(p)
    # sysC=syscost(550*1000)
    # print(sysC.total_embodied()[1])
    # print(sysC.energy())
    # cost = costet("input_example/dc.params", "input_example/foreio_r.params", model_file = 'textx.txt')
    # manu_bldg = eeio('Y_building_550.csv')
    # server = servers(550)
    # server.get_server_Y()
    # cns = eeio('Mix-SSD_550.csv')
    # bld = buildings(550)
    # bld.get_building(out_csv=True)
    # cns = eeio('Y_cns.csv')
    # print(cns.env_inventory()[1])
    # print(cns.emission_emp_impacts()[1])
    # print('\n')
    # print(manu_bldg.env_inventory()[1])
    # print(manu_bldg.emission_emp_impacts()[1])
    #print(util_bldg.direct_costs())
    # print(cns.total_costs().head())
    # cns.activity().to_csv('economic_sectors.csv')
    # print(cns.direct_costs())
    # print(manu_bldg.direct_costs())
    # print(cns.econ_impacts().head(10))
    # print(server.server_power(summed=True))
    # print(server.get_servercosts())
    # print(server.server_budget())
    # print(server.get_server_Y())
    # print(cns.env_inventory()[1])
    # print(manu_bldg.emission_emp_impacts()[1])
    # print(manu_bldg.TRACI_characterization_factor['air/unspecified/dinitrogen monoxide/kg'])
    # print(manu_bldg.TRACI())
    # print(manu_bldg.TRACI_impacts_of_selected_effect())
    # print(health_care.level_sectors('direct'))
    # print(util_bldg.activity().iloc[28])
    # print(manu_bldg.input()[1])
    # print(manu_bldg.input_building())
    # print(get_matrix())
    # print(util_bldg.input()[0])
    # print(manu_bldg.Matrix_A().loc['233230/manufacturing buildings/us'])
    # print()
    # print(manu_bldg.Matrix_A()['233230/manufacturing buildings/us'].loc['233230/manufacturing buildings/us'])
    # cost.view_raw_output(save = True)
    # cost.view_script('cost-et.pl')
    # print(cost.get_dc_params())
    # print(cost.get_resource_params())
    # print(cost.get_server_age())
    # print(cost.get_server_replacement())
    # print(cost.get_dc_costs())
    # cost.plot_dc_costs('test')
    # cost.get_dc_tco()
    # cost.res_dims()
    # cost.sys_cost_D('en',p = True)
    # cost.tau_curves('en')
    # cost.get_lang_tau('en','Ashburn')
    # cost.get_cap_lang_dc('zh')
    # print(pd.read_csv('sectors.csv').Sector.to_list())
    

if __name__ == "__main__":
  main()
