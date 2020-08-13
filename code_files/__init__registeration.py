from gym.envs.registration import register
import os
import fileinput

FD = os.path.dirname(os.path.realpath(__file__));

register(
    id='Eplus-demo-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/', # The EnergyPlus software path
            'weather_path':FD + '/envs/weather/pittsburgh_TMY3.epw', # The epw weather file
            'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
            'variable_path':FD + '/envs/eplus_models/demo_5z/learning/cfg/variables_v0.cfg', # The cfg file
            'idf_path':FD + '/envs/eplus_models/demo_5z/learning/idf/5ZoneAutoDXVAV_v0.idf', # The idf file
            'env_name': 'Eplus-demo-v1',
            })

register(
    id='Eplus-dc_golden-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/', # The EnergyPlus software path
            'weather_path':FD + '/envs/weather/USA_CO_Golden-NREL.724666_TMY3.epw', # The epw weather file
            'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
            'variable_path':FD + '/envs/eplus_models/datacenters/cfg/variables_v1.cfg', # The cfg file
            'idf_path':FD + '/envs/eplus_models/datacenters/idf/1ZD_CRAC_wPumpedDXCoolingCoil_golden.idf', # The idf file
            'env_name': 'Eplus-dc_golden-v0',
            })

register(
    id='Eplus-dc_Singapore-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/', # The EnergyPlus software path
            'weather_path':FD + '/envs/weather/SGP_Singapore_486980_IWEC.epw', # The epw weather file
            'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
            'variable_path':FD + '/envs/eplus_models/datacenters/cfg/variables_v2.cfg', # The cfg file
            'idf_path':FD + '/envs/eplus_models/datacenters/idf/2ZoneDataCenterHVAC_wEconomizer_RL_git.idf', # The idf file
            'env_name': 'Eplus-dc_Singapore-v0',
            })

register(
    id='Eplus-dc_Singapore-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-9-0/', # The EnergyPlus software path
            'weather_path':FD + '/envs/weather/SGP_Singapore_486980_IWEC.epw', # The epw weather file
            'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
            'variable_path':FD + '/envs/eplus_models/datacenters/cfg/variables_v2.cfg', # The cfg file
            'idf_path':FD + '/envs/eplus_models/datacenters/idf/eplus89/2ZoneDataCenterHVAC_wEconomizer_RL_git_86to89.idf', # The idf file
            'env_name': 'Eplus-dc_Singapore-v1',
            })

register(
    id='Eplus-dc_Ashburn-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/', # The EnergyPlus software path
            'weather_path':FD + '/envs/weather/USA_VA_Arlington-Ronald.Reagan.Washington.Natl.AP.724050_TMY3.epw', # The epw weather file
            'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
            'variable_path':FD + '/envs/eplus_models/datacenters/cfg/variables_v2.cfg', # The cfg file
            'idf_path':FD + '/envs/eplus_models/datacenters/idf/2ZoneDataCenterHVAC_wEconomizer_RL_git.idf', # The idf file
            'env_name': 'Eplus-dc_Ashburn-v0',
            })

register(
    id='Eplus-dc_Ashburn-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-9-0/', # The EnergyPlus software path
            'weather_path':FD + '/envs/weather/USA_VA_Arlington-Ronald.Reagan.Washington.Natl.AP.724050_TMY3.epw', # The epw weather file
            'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
            'variable_path':FD + '/envs/eplus_models/datacenters/cfg/variables_v2.cfg', # The cfg file
            'idf_path':FD + '/envs/eplus_models/datacenters/idf/eplus89/2ZoneDataCenterHVAC_wEconomizer_RL_git_86to89.idf', # The idf file
            'env_name': 'Eplus-dc_Ashburn-v1',
            })

register(
    id='Eplus-dc_SanFrancisco-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/', # The EnergyPlus software path
            'weather_path':FD + '/envs/weather/USA_CA_San.Francisco.724940_TMY2.epw', # The epw weather file
            'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
            'variable_path':FD + '/envs/eplus_models/datacenters/cfg/variables_v2.cfg', # The cfg file
            'idf_path':FD + '/envs/eplus_models/datacenters/idf/2ZoneDataCenterHVAC_wEconomizer_RL_git.idf', # The idf file
            'env_name': 'Eplus-dc_SanFrancisco-v0',
            })

register(
    id='Eplus-dc_SanFrancisco-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-9-0/', # The EnergyPlus software path
            'weather_path':FD + '/envs/weather/USA_CA_San.Francisco.724940_TMY2.epw', # The epw weather file
            'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
            'variable_path':FD + '/envs/eplus_models/datacenters/cfg/variables_v2.cfg', # The cfg file
            'idf_path':FD + '/envs/eplus_models/datacenters/idf/eplus89/2ZoneDataCenterHVAC_wEconomizer_RL_git_86to89.idf', # The idf file
            'env_name': 'Eplus-dc_SanFrancisco-v1',
            })

register(
    id='Eplus-dc_Haarlem-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/', # The EnergyPlus software path
            'weather_path':FD + '/envs/weather/NLD_Amsterdam.062400_IWEC.epw', # The epw weather file
            'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
            'variable_path':FD + '/envs/eplus_models/datacenters/cfg/variables_v2.cfg', # The cfg file
            'idf_path':FD + '/envs/eplus_models/datacenters/idf/2ZoneDataCenterHVAC_wEconomizer_RL_git.idf', # The idf file
            'env_name': 'Eplus-dc_Haarlem-v0',
            })

register(
    id='Eplus-dc_Haarlem-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-9-0/', # The EnergyPlus software path
            'weather_path':FD + '/envs/weather/NLD_Amsterdam.062400_IWEC.epw', # The epw weather file
            'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
            'variable_path':FD + '/envs/eplus_models/datacenters/cfg/variables_v2.cfg', # The cfg file
            'idf_path':FD + '/envs/eplus_models/datacenters/idf/eplus89/2ZoneDataCenterHVAC_wEconomizer_RL_git_86to89.idf', # The idf file
            'env_name': 'Eplus-dc_Haarlem-v1',
            })

register(
    id='Eplus-dc_Carrollton-v0',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/', # The EnergyPlus software path
            'weather_path':FD + '/envs/weather/USA_TX_Dallas-Addison.AP.722598_TMY3.epw', # The epw weather file
            'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
            'variable_path':FD + '/envs/eplus_models/datacenters/cfg/variables_v2.cfg', # The cfg file
            'idf_path':FD + '/envs/eplus_models/datacenters/idf/2ZoneDataCenterHVAC_wEconomizer_RL.idf', # The idf file
            'env_name': 'Eplus-dc_Carrolton-v0',
            })
    
register(
    id='Eplus-dc_Carrollton-v1',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-9-0/', # The EnergyPlus software path
            'weather_path':FD + '/envs/weather/USA_TX_Dallas-Addison.AP.722598_TMY3.epw', # The epw weather file
            'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
            'variable_path':FD + '/envs/eplus_models/datacenters/cfg/variables_v2.cfg', # The cfg file
            'idf_path':FD + '/envs/eplus_models/datacenters/idf/eplus89/2ZoneDataCenterHVAC_wEconomizer_RL_git_86to89.idf', # The idf file
            'env_name': 'Eplus-dc_Carrolton-v1',
            })

register(
    id='Eplus-demo-v92',
    entry_point='eplus_env.envs:EplusEnv',
    kwargs={'eplus_path':FD + '/envs/EnergyPlus-9-2-0/', # The EnergyPlus software path
            'weather_path':FD + '/envs/weather/USA_CO_Golden-NREL.724666_TMY3.epw', # The epw weather file
            'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
            'variable_path':FD + '/envs/eplus_models/datacenters/cfg/variables_v1.cfg', # The cfg file
            'idf_path':FD + '/envs/eplus_models/demo_5z/learning/idf/eplus92_5ZoneAutoDXVAV.idf', # The idf file
            'env_name': 'Eplus-demo-v92',
            });

