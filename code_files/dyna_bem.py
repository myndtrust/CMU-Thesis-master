import gym;
import eplus_env;
import pandas as pd
import numpy as np
from termcolor import colored
import streamlit as st
#import tensorflow as tf
import random
import sys
sys.path.insert(0, "/home/eric/ramukcire/estimating_cost_of_dc_services/syscost/")
import traffic.traffic as traffic
#print(colored((sys.path), 'red'))

#print('TF Version: '+str(tf.__version__))

class agent(object):
    def __init__(self, env, lang, traffic_file=None):
        self.env = gym.make(env)
        self.lang = lang
        self.traffic = traffic_file
        self.observations = ['HTSP','CTSP', 'ITE_Power[w]','Total-Facility_Power[w]', 'Facility-HVAC_Demand_power[w]']
        self.Time = []
        self.Obs = []
        self.Term = []
        self.Episode = []
        self.Site = []
        self.CPU = []
        self.Sites = ['Ashburn','Carrollton','Haarlem','San Francisco','Singapore']

    def get_action(self):
        actions = [[0,25]]
        choice = random.randint(0,len(actions)-1)
        action = actions[choice]
        return list(action)

    def _traffic(self):
        '''Result indicates Normalized daily traffic statistics.'''
        tau_dict = {}
        tau = traffic.traffic(self.traffic, 0.50)

        for site in self.Sites:
            tau_dict[site] = tau.tau()[str(site)+' '+self.lang][-365:]

        return tau_dict

    def expand_traffic(self):
        '''Modulo expansion form daily to hourly traffic'''
        dmd = {}
        for site in self.Sites:
            traffic_list = self._traffic()[site]
            init_val = traffic_list[0]
            dmd[site]=[init_val]
            for i in range(1,8760):
                dmd[site].append(traffic_list[int(i/24)])
        return dmd

    def get_traffic(self, site, i):
        traffic = self.expand_traffic()[site][i]
        return type([traffic])

    def schedule_run(self, site, dmd, no_runs=1):
        for episode in range(no_runs):
            curSimTime, ob, isTerminal = self.env.reset(); # Reset the env (create the EnergyPlus subprocess)
            self.Episode.append(episode)
            self.Site.append(site)
            self.Time.append(int(curSimTime))
            self.Obs.append([round(i,2) for i in ob])
            self.Term.append(isTerminal)
            self.CPU.append(0)
            # traffic = self.expand_traffic()[site]
            traffic = dmd[site]
            i = 0
            while not isTerminal:
                action = self.get_action() # (HtgSetP-RL, ClgSetP-RL)
                # print('Temps: ',action, '.Temps type: ', type(action))
                cpu_load = np.round(traffic[i], 2)
                #cpu_load = 1
                # print('Cpu load: ',cpu_load, '.Type cpu_load: ', type(cpu_load))
                action.append(cpu_load)
                # print('Action: ', action, 'Type Action: ', type(action))
                curSimTime, ob, isTerminal = self.env.step(action)
                self.Episode.append(episode)
                self.Site.append(site)
                self.Time.append(int(curSimTime))
                self.CPU.append(cpu_load)
                self.Obs.append([round(i,2) for i in ob])
                self.Term.append(isTerminal)
                i = i+1

        self.env.end_env(); # Safe termination of the environment after use. 

        dict ={'Cur_Episode': self.Episode,'Site': self.Site, 'Time_t': self.Time,'CPU_load': self.CPU,'Is_terminal': self.Term}
        df = pd.DataFrame(dict)
        df1 = pd.DataFrame(self.Obs, columns=self.observations)
        con_df = pd.concat([df,df1], axis =1)
        st.table(con_df)
        print(con_df)
        return con_df

def main():
    traffic_file_1 = '../../traffic/data/unit_testing.csv'
    traffic_file_2 = '../../traffic/data/train_1.csv'
   
    # Agent = agent('Eplus-dc_golden-v0','en','../../traffic/data/unit_testing.csv')
    
    # singapore = agent('Eplus-dc_Singapore-v0','en',traffic_file_1)
    # dmd = singapore.expand_traffic()
    # singapore.schedule_run('Singapore', dmd=dmd)
    # print(dmd)

    # ashburn = agent('Eplus-dc_Ashburn-v0','en', traffic_file_2)
    # dmd = ashburn.expand_traffic()
    # ashburn.schedule_run('Ashburn', dmd=dmd)

    # carrolton = agent('Eplus-dc_Carrollton-v0','en', traffic_file_2)
    # dmd = carrolton.expand_traffic()
    # carrolton.schedule_run('Carrollton', dmd=dmd)

    # haarlem = agent('Eplus-dc_Haarlem-v0','en',traffic_file_2)
    # dmd = haarlem.expand_traffic()
    # print(haarlem.schedule_run('Haarlem', dmd=dmd))

    sfo = agent('Eplus-dc_SanFrancisco-v0','en',traffic_file_1)
    dmd = sfo.expand_traffic()
    # print(sfo.schedule_run('San Francisco', dmd=dmd))
    print(dmd)


    #Agent.run()
    #print(Agent.get_action())
    #print((Agent.expand_traffic()['San Francisco'][0:26]))
    #print((Agent._traffic()))
    #print((Agent._traffic()['Ashburn']))
    #print(Agent._traffic('en')['San Francisco'][40])
    #print('The traffic list has '+str(len(Agent._traffic('en')['San Francisco'])) + ' days.')
    #print(Agent.get_traffic('Ashburn', 0))
    #test = traffic.traffic(traffic_file, 0.95)
    # test.information()

if __name__ == "__main__":
  main()

