
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import math

import os.path



import numpy as np

import pandas as pd


from pysc2.agents import base_agent,EA

from pysc2.env import sc2_env



from pysc2.lib import actions, features, units
import logging
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
class ScLogger:

    logger = logger = logging.getLogger()



    @staticmethod

    def log(message):

        ScLogger.logger.info("[SC LOGGER] : %s", message)



    @staticmethod

    def logAgent(message):

        ScLogger.logger.info("[SC LOGGER][AGENT] : %s", message)



    @staticmethod

    def logbo(message):

        ScLogger.logger.info("[SC LOGGER][BUILD ORDER] : %s", message)



    @staticmethod

    def logReward(message):

        ScLogger.logger.info("[SC LOGGER][REWARD] : %.2f", message)
_VESPENE_GEYSER = 342
_NEUTRAL_MINERAL_FIELD = 341
BANELING = 9
BANELINGBURROWED = 115
BANELINGCOCOON = 8
BANELINGNEST = 96
BROODLORD = 114
BROODLORDCOCOON = 113
BROODLING = 289
BROODLINGESCORT = 143
CHANGELING = 12
CHANGELINGMARINE = 15
CHANGELINGMARINESHIELD = 14
CHANGELINGZEALOT = 13
CHANGELINGZERGLING = 17
CHANGELINGZERGLINGWINGS = 16
CORRUPTOR = 112
CREEPTUMOR = 87
CREEPTUMORBURROWED = 137
CREEPTUMORQUEEN = 138
DRONE = 104
DRONEBURROWED = 116
COCOON = 103
EVOLUTIONCHAMBER = 90
EXTRACTOR = 88
GREATERSPIRE = 102
HATCHERY = 86
HIVE = 101
HYDRALISK = 107
HYDRALISKBURROWED = 117
HYDRALISKDEN = 91
INFESTATIONPIT = 94
INFESTEDTERRAN = 7
INFESTEDTERRANBURROWED = 120
INFESTEDTERRANCOCOON = 150
INFESTOR = 111
INFESTORBURROWED = 127
LAIR = 100
LARVA = 151
LOCUST = 489
LOCUSTFLYING = 693
LURKER = 502
LURKERBURROWED = 503
LURKERDEN = 504
LURKERCOCOON = 501
MUTALISK = 108
NYDUSCANAL = 142
NYDUSNETWORK = 95
OVERLORD = 106
OVERLORDTRANSPORT = 893
OVERLORDTRANSPORTCOCOON = 892
OVERSEER = 129
OVERSEERCOCOON = 128
OVERSEERONSIGHTMODE = 1912
QUEEN = 126
QUEENBURROWED = 125
RAVAGER = 688
RAVAGERBURROWED = 690
RAVAGERCOCOON = 687
ROACH = 110
ROACHBURROWED = 118
ROACHWARREN = 97
SPAWN = 89
SPINECRAWLER = 98
SPINCRAWLERUPROOTED = 139
SPIRE = 92
SPORECRAWLER = 99
SPRECRAWLERUPROOTED = 140
SWARMHOST = 494
SWARMHOSTBURROWED = 493
ULTRALISK = 109
ULTRALISKBURROWED = 131
ULTRALISKCAVERN = 93
VIPER = 499
ZERGLING = 105
ZERGLINGBURROWED = 119
_SMART_SCREEN = actions.FUNCTIONS.Smart_screen.id
_BUILD_BANELINGNEST = actions.FUNCTIONS.Build_BanelingNest_screen.id
_BUILD_CREEPTUMOR = actions.FUNCTIONS.Build_CreepTumor_screen.id
_BUILD_EVOCHAMBER = actions.FUNCTIONS.Build_EvolutionChamber_screen.id
_BUILD_HYDRALISKDEN = actions.FUNCTIONS.Build_HydraliskDen_screen.id
_BUILD_INFESTATIONPIT = actions.FUNCTIONS.Build_InfestationPit_screen.id
_BUILD_LURKERDEN = actions.FUNCTIONS.Build_LurkerDen_screen.id
_BUILD_NYDUSWORK = actions.FUNCTIONS.Build_NydusWorm_screen.id
_BUILD_NYDUSNETWORK = actions.FUNCTIONS.Build_NydusNetwork_screen.id
_BUILD_ROACHWARREN = actions.FUNCTIONS.Build_RoachWarren_screen.id
_BUILD_SPINECRAWLER = actions.FUNCTIONS.Build_SpineCrawler_screen.id
_BUILD_SPIRE = actions.FUNCTIONS.Build_Spire_screen.id
_BUILD_SPORECRAWLER = actions.FUNCTIONS.Build_SporeCrawler_screen.id
_BUILD_ULTRALISKCAVERN = actions.FUNCTIONS.Build_UltraliskCavern_screen.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_MAKE_LARVA = actions.FUNCTIONS.Effect_InjectLarva_screen.id
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_BUILD_EXTRACTOR = actions.FUNCTIONS.Build_Extractor_screen.id
_BURROW = actions.FUNCTIONS.BurrowDown_quick.id
_BURROW_UP = actions.FUNCTIONS.BurrowUp_quick.id
_CANCEL = actions.FUNCTIONS.Cancel_quick.id
_MORPH_LAIR = actions.FUNCTIONS.Morph_Lair_quick.id
_MORPH_HIVE = actions.FUNCTIONS.Morph_Hive_quick.id
_MORPH_LURKER = actions.FUNCTIONS.Morph_Lurker_quick.id
_MORPH_LURKERDEN = actions.FUNCTIONS.Morph_LurkerDen_quick.id
_MORPH_OVERSEER = actions.FUNCTIONS.Morph_Overseer_quick.id
_MORPH_GREATERSPIRE = actions.FUNCTIONS.Morph_GreaterSpire_quick.id
_MORPH_ROOT = actions.FUNCTIONS.Morph_Root_screen.id
_MORPH_RAVAGER = actions.FUNCTIONS.Morph_Ravager_quick.id
_MOVE = actions.FUNCTIONS.Move_screen.id
_RALLY_WORKERS = actions.FUNCTIONS.Rally_Hatchery_Workers_screen.id
_RESEARCH_BURROW = actions.FUNCTIONS.Research_Burrow_quick.id
_TRAIN_BANELING = actions.FUNCTIONS.Train_Baneling_quick.id
_TRAIN_CORRUPTER = actions.FUNCTIONS.Train_Corruptor_quick.id
_TRAIN_HYDRALISK = actions.FUNCTIONS.Train_Hydralisk_quick.id
_TRAIN_INFESTOR = actions.FUNCTIONS.Train_Infestor_quick.id
_TRAIN_MUTALISK = actions.FUNCTIONS.Train_Mutalisk_quick.id
_TRAIN_OBSERVER = actions.FUNCTIONS.Train_Observer_quick.id
_TRAIN_ROACH = actions.FUNCTIONS.Train_Roach_quick.id
_TRAIN_SWARMHOST = actions.FUNCTIONS.Train_SwarmHost_quick.id
_TRAIN_ULTRALISK = actions.FUNCTIONS.Train_Ultralisk_quick.id
_TRAIN_BROODLORD = actions.FUNCTIONS.Morph_BroodLord_quick.id
_GENERATE_CREEP_ON = actions.FUNCTIONS.Behavior_GenerateCreepOn_quick.id
_GENERATE_CREEP_OFF = actions.FUNCTIONS.Behavior_GenerateCreepOff_quick.id
_EXPLODE = actions.FUNCTIONS.Effect_Explode_quick.id
_LOAD = actions.FUNCTIONS.Load_screen.id
_CONSUME = actions.FUNCTIONS.Effect_ViperConsume_screen.id
_SPAWN_LOCUSTS = actions.FUNCTIONS.Effect_SpawnLocusts_screen.id
_SPAWN_CHANGELING = actions.FUNCTIONS.Effect_SpawnChangeling_quick.id
_SWOOP = actions.FUNCTIONS.Effect_LocustSwoop_screen.id
_MORPH_OVERSIGHT = actions.FUNCTIONS.Morph_OversightMode_quick.id
_RESEARCH_TALONS = actions.FUNCTIONS.Research_AdaptiveTalons_quick.id
_RESEARCH_HOOKS = actions.FUNCTIONS.Research_CentrifugalHooks_quick.id
_RESEARCH_CLAWS = actions.FUNCTIONS.Research_DrillingClaws_quick.id
_RESEARCH_SPINES = actions.FUNCTIONS.Research_GroovedSpines_quick.id
_RESEARCH_GLANDS = actions.FUNCTIONS.Research_PathogenGlands_quick.id
_RESEARCH_TUNNELCLAWS = actions.FUNCTIONS.Research_TunnelingClaws_quick.id
_RESEARCH_ZERG_FLYER_ARMOR = actions.FUNCTIONS.Research_ZergFlyerArmor_quick.id
_RESEARCH_ZERG_FLYER_ATTACK = actions.FUNCTIONS.Research_ZergFlyerAttack_quick.id
_RESEARCH_ZERG_GROUND_ARMOR = actions.FUNCTIONS.Research_ZergGroundArmor_quick.id
_RESEARCH_ZERG_MELEE_WEAPONS = actions.FUNCTIONS.Research_ZergMeleeWeapons_quick.id
_RESEARCH_ZERG_MISSILE_WEAPONS = actions.FUNCTIONS.Research_ZergMissileWeapons_quick.id
_RESEARCH_ADRENAL_GLANDS = actions.FUNCTIONS.Research_ZerglingAdrenalGlands_quick.id
_RESEARCH_METABOLIC_BOOST = actions.FUNCTIONS.Research_ZerglingMetabolicBoost_quick.id
_RALLY_UNITS = actions.FUNCTIONS.Rally_Units_screen.id
_CREEP_TUMOR = actions.FUNCTIONS.Build_CreepTumor_screen.id
_CORROSIVEBILE = actions.FUNCTIONS.Effect_CorrosiveBile_screen.id
_CONTAMINATE = actions.FUNCTIONS.Effect_Contaminate_screen.id
_FUNGALGROWTH = actions.FUNCTIONS.Effect_FungalGrowth_screen.id
_INFESTEDTERRANS = actions.FUNCTIONS.Effect_InfestedTerrans_screen.id
_CAUSTISPRAY = actions.FUNCTIONS.Effect_CausticSpray_screen.id
_ABDUCT = actions.FUNCTIONS.Effect_Abduct_screen.id
_BLINDCLOUD = actions.FUNCTIONS.Effect_BlindingCloud_screen.id
_PARASITICBOMB = actions.FUNCTIONS.Effect_ParasiticBomb_screen.id
_RESEARCH_GLIAL_RECONSTITUTION = actions.FUNCTIONS.Research_GlialRegeneration_quick.id
_TRANSFUSION = actions.FUNCTIONS.Effect_Transfusion_screen.id
_RESEARCH_PNEUMATIZED_CARAPACE = actions.FUNCTIONS.Research_PneumatizedCarapace_quick.id
_RESEARCH_MUSCULAR_AUGMENTS = actions.FUNCTIONS.Research_MuscularAugments_quick.id
_RESEARCH_CHITINOUS_PLATING = actions.FUNCTIONS.Research_ChitinousPlating_quick.id
_TRAIN_VIPER = actions.FUNCTIONS.Train_Viper_quick.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_RESEARCH_NEURAL_PARASITE = actions.FUNCTIONS.Research_NeuralParasite_quick.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]
obs = None
_NO_OP = actions.FUNCTIONS.no_op.id

_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_BUILD_OVERLORD = actions.FUNCTIONS.Train_Overlord_quick.id
_BUILD_SPAWN = actions.FUNCTIONS.Build_SpawningPool_screen.id
_BUILD_HATCHERY = actions.FUNCTIONS.Build_Hatchery_screen.id
_BUILD_DRONE = actions.FUNCTIONS.Train_Drone_quick.id
_BUILD_QUEEN = actions.FUNCTIONS.Train_Queen_quick.id

_BUILD_ZERGLING = actions.FUNCTIONS.Train_Zergling_quick.id

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id

_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_CREEP = features.SCREEN_FEATURES.player_relative.index

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

wins = 0

_PLAYER_SELF = 1

_PLAYER_HOSTILE = 4

_ARMY_SUPPLY = 5




ACTION_DO_NOTHING = 'donothing'

ACTION_BUILD_SPAWN = 'buildspawn'

ACTION_BUILD_DRONE = 'builddrone'
ACTION_BUILD_OVERLORD = 'buildoverlord'
ACTION_BUILD_HATCHERY = 'buildhatchery'
ACTION_BUILD_ZERGLING = 'buildzergling'
ACTION_ATTACK = 'attack'
ACTION_BUILD_QUEEN = 'buildqueen'
ACTION_MOVE_CAMERA = 'movecamera'
ACTION_HARVEST_MINERAL = 'harvestmineral'
ACTION_MAKE_LARVA = 'makelarva'
ACTION_BUILD_EXTRACTOR = 'buildextractor'
ACTION_MINE_GAS = 'minegas'
ACTION_BUILD_BANELING_NEST = 'buildganelingnest'
ACTION_BUILD_CREEP_TUMOR = 'buildcreeptumor'
ACTION_BUILD_HYDRALISKDEN = 'buildhydraliskden'
ACTION_MORPH_LAIR = 'morphlair'
ACTION_MORPH_HIVE = 'morphhive'
ACTION_BUILD_INFESTOR = 'buildinfestor'
ACTION_BUILD_HYDRALISK = 'buildhydralisk'
ACTION_BUILD_INFESTATIONPIT = 'buildinfestationpit'
ACTION_BUILD_ULTRALISK = 'buildultralisk'
ACTION_BUILD_EVOCHAMBER = 'buildevochamber'
ACTION_BUILD_ULTRALISKCAVERN = 'buildultraliskcavern'
ACTION_BUILD_ROACHWARREN = 'buildroachwarren'
ACTION_MORPH_OVERSEER = 'morphoverseer'
ACTION_BUILD_ROACH = 'buildroach'
ACTION_BUILD_SPORECRAWLER = 'buildsporecrawler'
ACTION_BUILD_SPINECRAWLER = 'buildspinecrawler'
ACTION_BUILD_BANELING = 'buildbaneling'

ACTION_BURROW_UP = 'burrowup'
ACTION_MOVE = 'move'
ACTION_CANCEL = 'cancel'
ACTION_RESEARCH_BURROW = 'researchburrow'
ACTION_RALLY_WORKERS = 'rallyworkers'
ACTION_MORPH_RAVAGER = 'morphravager'
ACTION_MORPH_ROOT = 'morphroot'
ACTION_MORPH_LURKERDEN = 'morphlurkerden'
ACTION_MORPH_LURKER = 'morphlurker'
ACTION_BUILD_LURKERDEN = 'buildlurkerden'
ACTION_BUILD_NYDUSWORK = 'buildnyduswork'
ACTION_BUILD_SPIRE = 'buildspire'
ACTION_BUILD_NYDUSNETWORK = 'buildnydusnetwork'
ACTION_BURROW = 'burrow'
ACTION_BUILD_SWARMHOST = 'buildswarmhost'
ACTION_BUILD_MUTALISK = 'buildmutalisk'
ACTION_BUILD_CORRUPTER = 'buildcorrupter'
ACTION_BUILD_BROODLORD = 'buildbroodlord'
ACTION_BUILD_GREATERSPIRE = 'buildgreaterspire'
reward = 0
ACTION_RESEARCH_TUNNELCLAWS = 'researchtunnelclaws'
ACTION_RESEARCH_FLYING_ARMOR = 'researchflyingarmor'
ACTION_RESEARCH_FLYING_ATTACK = 'researchflyingattack'
ACTION_RESEARCH_GROUND_ARMOR = 'researchgroundarmor'
ACTION_RESEARCH_MELEE_WEAPONS = 'researchmeleeweapons'
ACTION_RESEARCH_MISSLE_WEAPONS = 'researchmissleweapons'
ACTION_RESEARCH_ADRENAL_GLANDS = 'researchadrenalglands'
ACTION_RESEARCH_METABOLIC_BOOST = 'researchmetabolicboost'
ACTION_BUILD_RAVAGER = 'buildravager'
ACTION_BUILD_VIPER = 'buildviper'
ACTION_EXPLODE = 'explode'
ACTION_EXPLODE_BUILDING = 'explodebuilding'
ACTION_EXPLODE_BUILDING_OFF = 'explodebuildingoff'
ACTION_LOAD = 'load'
ACTION_CONSUME = 'consume'
ACTION_SPAWN_LOCUSTS = 'spawnlocusts'
ACTION_CHANGELING = 'changeling'
ACTION_SWOOP = 'swoop'
ACTION_MORPH_OVERSIGHT = 'morphoversight'
ACTION_RALLY = 'rally'
ACTION_CREEP_TUMOR = 'creeptumor'
ACTION_GENERATE_CREEP_ON = 'generatecreepon'
ACTION_GENERATE_CREEP_OFF = 'generatecreepoff'
ACTION_ROOT = 'root'
ACTION_TRANSFUSION = 'transfusion'
ACTION_RESEARCH_CENTRIFUGAL_HOOKS = 'researchcentrifugalhooks'
ACTION_RESEARCH_GLIAL_RECONSTITUTION = 'researchgilialreconstitution'
ACTION_RESEARCH_TUNNELING_CLAWS = 'researchtunnelingclaws'
ACTION_CORROSIVE_BILE = 'corrosivebile'
ACTION_RESEARCH_GROOVED_SPINES = 'researchgroovedspines'
ACTION_RESEARCH_MUSCULAR_AUGMENTS = 'researchmuscularaugments'
ACTION_RESEARCH_ADAPTIVE_TALONS = 'researchadaptivetalons'
ACTION_NEURAL_PARASITE = 'neuralparasite'
ACTION_FUNGAL_GROWTH = 'fungalgrowth'
ACTION_INFESTED_TERRAN ='infestedterran'
ACTION_RESEARCH_NEURAL_PARASITE = 'researchneuralparasite'
ACTION_RESEARCH_PATHOGEN_GLANDS = 'researchpathogenglands'
ACTION_RESEARCH_CHITINOUS_PLATING = 'researchchitinousplating'
ACTION_RESEARCH_PNEUMATIZED_CARAPACE = 'researchpneumatizedcarapace'
ACTION_BLINDING_CLOUD = 'blindingcloud'
ACTION_ABDUCT = 'abduct'
ACTION_MOVE_SCREEN = 'movescreen'
ACTION_PARASITIC_BOMB = 'parasiticbomb'
counter = 0

class QLearningTable:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay

        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


    def choose_action(self, observation):

        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:

            # choose best action

            state_action = self.q_table.ix[observation, :]

            

            # some actions have the same value

            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:

            # choose random action

            action = np.random.choice(self.actions)

            

        return action



    def learn(self, s, a, r, s_):
        if s == s_:
            return
        self.check_state_exist(s_)

        self.check_state_exist(s)

        

        q_predict = self.q_table.ix[s, a]


        if s_ != 'terminal':

            q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        else:

            q_target = r  # next state is terminal

            

        # update

        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)



    def check_state_exist(self, state):

        if state not in self.q_table.index:

            # append new state to q table

            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class ZergAgent(base_agent.BaseAgent):
  smart_actions = [0]
  DATA_FILE = ''		
  base_top_left = None
  larva = False
  supply_depot_built = False
  scv_selected = False
  barracks_built = False

  barracks_selected = False

  barracks_rallied = False

  army_selected = False

  army_rallied = False
  NO_OP = None

  SELECT_POINT = None

  TRAIN_OVERLORD = None

  BUILD_SPAWN = None

  TRAIN_ZERGLING = None

  ATTACK_MINIMAP = None  

  HARVEST_GATHER = None
  RALLY_UNITS_MINIMAP = None
  _PLAYER_RELATIVE = None
  _UNIT_TYPE = None
  fails = 0
  counter2 = 0
  succeeds = 0
  def __init__(self):

    super(ZergAgent, self).__init__()
    self.attack_coordinates = None
    self.qlearn = QLearningTable(actions=list(range(EA.DNA_SIZE)))
    self.previous_action = None
    self.previous_state = None
    self.obs = None
    self.h_y = None
    self.h_x = None
    self.move_number = 0
    self.counter2 = 0
    self.counter = 0
    self.reward = 0
    if os.path.isfile(self.DATA_FILE + '.gz'):
        self.qlearn.q_table = pd.read_pickle(self.DATA_FILE + '.gz', compression='gzip')
  def transformLocation(self, x, y):
      return [x, y]		
  def reward():
      number = (obs.observation['score_cumulative'][_PLAYER_RELATIVE] - obs.observation['score_cumulative'][_PLAYER_HOSTILE] - fails)
      self.reward = obs.reward - (obs.reward * (abs(number)/(1+abs(number))))
		  
  def splitAction(self, action_id):
        
        smart_action = self.smart_actions[action_id]
		
        x = 0
        y = 0
        if '_' in smart_action:
             smart_action, x, y = smart_action.split('_')
        return(smart_action,x,y)
		
  def set_actions(self,action_list):
    if len(self.smart_actions) > 0:
      self.smart_actions.clear()
    for i in range(len(action_list)):
      self.smart_actions.append(action_list[i])
  def set_datafile(self,filename):
    self.DATA_FILE = filename
	
  def findLocationForBuilding(self,action):
     x = random.randint(0,83)
     y = random.randint(0,83)
     target = [x,y]
     try:
        actions.FunctionCall(action,_NOT_QUEUED)
     except ValueError:
        return 0
     return target
	 
  def can_do(self, obs, action):

    return action in obs.observation.available_actions

  def step(self, obs,DATA_FILE):
    self.obs = obs
    super(ZergAgent, self).step(obs)
    self.counter = self.counter + len(obs.observation['last_actions'])
    self.counter2 += 1
    succeeds = 0
    fails = 0
    if obs.last():
        number = (obs.observation['score_cumulative'][_PLAYER_RELATIVE] - obs.observation['score_cumulative'][_PLAYER_HOSTILE])
        if(number < 0):
           self.reward =(obs.reward * ((abs(number)/(1+abs(number)))))
        else:
           self.reward = (obs.reward *((number/(1+number))))
        print(self.counter)
        self.reward = ((self.counter/self.counter2) + ((self.counter/self.counter2) * self.reward)) + obs.reward
        self.qlearn.learn(str(self.previous_state), self.previous_action, self.reward, 'terminal')
        self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
        self.previous_action = None
        self.previous_state = None
        self.move_number = 0
        return actions.FunctionCall(_NO_OP,[])
		
    unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
    if obs.first():
        player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
    if self.base_top_left is None:
        player_y,player_x = (obs.observation["feature_minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        self.h_y, self.h_x = (unit_type == HATCHERY).nonzero()
    h_y, h_x = (unit_type == HATCHERY).nonzero()
    h_count = int(round(len(h_y)/69))
	
    o_y, o_x = (unit_type == units.Zerg.Overlord).nonzero()
    overlord_count = int(round(len(o_y)/ 69))
	
    s_y, s_x = (unit_type == SPAWN).nonzero()
    spawn_count = int(round(len(s_y)/69))
	
    l_y, l_x = (unit_type == LARVA).nonzero()
    larva_count = int(round(len(l_y)/69))
	
    d_y, d_x = (unit_type == DRONE).nonzero()
    drone_count = int(round(len(d_y)/69))
    
    e_y, e_x = (unit_type == EXTRACTOR).nonzero()
    extractor_count = int(round(len(e_y)/60))
	
    q_y, q_x = (unit_type == QUEEN).nonzero()
    queen_count = int(round(len(q_y)/69))
	
    m_y, m_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
    mineral_count = int(round(len(m_y)/69))
    creep_y, creep_x = (obs.observation['feature_screen'][_CREEP].nonzero())
    creep_count = len(creep_y)
	
    ba_y , ba_x = (unit_type == BANELINGNEST).nonzero()
    banelingnest_count = int(round(len(ba_y)/69))
	
    ev_y, ev_x = (unit_type == EVOLUTIONCHAMBER).nonzero()
    evochamber_count = int(round(len(ev_y)/69))
	
    hy_y, hy_x = (unit_type == HYDRALISKDEN).nonzero()
    hydraliskden_count = int(round(len(hy_y)/69))
	
    la_y, la_x = (unit_type == LAIR).nonzero()
    lair_count = int(round(len(la_y)/69))
	
    in_y, in_x = (unit_type == INFESTATIONPIT).nonzero()
    infestpit_count = int(round(len(in_y)/69))
	
    hi_y, hi_x = (unit_type == HIVE).nonzero()
    hive_count = int(round(len(hi_y)/69))
	
    ul_y, ul_x = (unit_type == ULTRALISKCAVERN).nonzero()
    ultraliskcav_count = int(round(len(ul_y)/69))
	
    ro_y, ro_x = (unit_type == ROACHWARREN).nonzero()
    roachwarren_count = int(round(len(ro_y)/69))
	
    sp_y, sp_x = (unit_type == SPORECRAWLER).nonzero()
    spore_count = int(round(len(sp_y)/69))
	
    spi_y, spi_x = (unit_type == SPINECRAWLER).nonzero()
    spine_count = int(round(len(spi_y)/69))

    ov_y, ov_x = (unit_type == OVERSEER).nonzero()
    overseer_count = int(round(len(ov_y)/69))

    spir_y, spir_x = (unit_type == SPIRE).nonzero()
    spire_count = int(round(len(spir_y)/69))
	
    spir_y, spir_x = (unit_type == LURKERDEN).nonzero()
    lurkerden_count = int(round(len(spir_y)/69))
	
    spir_y, spir_x = (unit_type == GREATERSPIRE).nonzero()
    greaterspire_count = int(round(len(spir_y)/69))
	
    spir_y, spir_x = (unit_type == NYDUSCANAL).nonzero()
    nydus_count = int(round(len(spir_y)/69))
	
    spir_y, spir_x = (unit_type == ZERGLING).nonzero()
    zergling_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == ROACH).nonzero()
    roach_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == HYDRALISK).nonzero()
    hydralisk_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == LURKER).nonzero()
    lurker_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == MUTALISK).nonzero()
    mutalisk_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == CORRUPTOR).nonzero()
    corrupter_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == BROODLORD).nonzero()
    broodlord_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == BANELING).nonzero()
    baneling_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == SWARMHOST).nonzero()
    swarmhost_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == INFESTOR).nonzero()
    infestor_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == ULTRALISK).nonzero()
    ultralisk_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == RAVAGER).nonzero()
    ravager_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == VIPER).nonzero()
    viper_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == CREEPTUMOR).nonzero()
    creep_tumor_count = int(round(len(spir_y)/69))	
    spir_y, spir_x = (unit_type == CHANGELING).nonzero()
    changeling_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == CHANGELINGMARINE).nonzero()
    changelingmarine_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == CHANGELINGMARINESHIELD).nonzero()
    changelingmarineshield_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == CHANGELINGZEALOT).nonzero()
    changelingzealot_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == CHANGELINGZERGLING).nonzero()
    changelingzergling_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == CHANGELINGZERGLINGWINGS).nonzero()
    changelingzerglingwings_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == LOCUST).nonzero()
    locust_count = int(round(len(spir_y)/69))
    spir_y, spir_x = (unit_type == BROODLING).nonzero()
    broodling_count = int(round(len(spir_y)/69))
	
    if self.move_number ==0:
        self.move_number +=1
        current_state = np.zeros(65)
        current_state[0] = broodling_count
        current_state[1] = locust_count
        current_state[2] = changeling_count
        current_state[3] = changelingmarine_count
        current_state[4] = changelingmarineshield_count
        current_state[5] = changelingzergling_count
        current_state[6] = changelingzerglingwings_count
        current_state[7] = changelingzealot_count
        current_state[8] = h_count
        current_state[9] = drone_count
        current_state[10] = overlord_count
        current_state[11] = obs.observation['player'][_ARMY_SUPPLY]
        current_state[12] = obs.observation['player'][1]
        current_state[13] = obs.observation['player'][10]
        current_state[14] = obs.observation['player'][7]
        current_state[15]= obs.observation['player'][4]
        current_state[16]= obs.observation['player'][5]
        current_state[17] = obs.observation['player'][3]
        current_state[18] = obs.observation['player'][2]
        current_state[19] = obs.observation['player'][0]
        current_state[20]= obs.observation['player'][6]
        current_state[21] = spawn_count
        current_state[22] = larva_count
        current_state[23] = creep_count
        current_state[24] = queen_count
        current_state[25] = extractor_count
        current_state[26] = mineral_count
        current_state[27] = hydraliskden_count
        current_state[28] = evochamber_count
        current_state[29] = banelingnest_count
        current_state[30] = ultraliskcav_count
        current_state[31] = infestpit_count
        current_state[32] = lair_count
        current_state[33] = hive_count
        current_state[34] = roachwarren_count
        current_state[35] = spore_count
        current_state[36] = spine_count
        current_state[37] = overseer_count
        current_state[38] = spire_count
        current_state[39] = lurkerden_count
        current_state[40] = greaterspire_count
        current_state[41] = nydus_count		
        current_state[42] = zergling_count
        current_state[43] = roach_count
        current_state[44] = mutalisk_count
        current_state[45] = hydralisk_count
        current_state[46] = lurker_count
        current_state[47] = broodlord_count
        current_state[48] = corrupter_count
        current_state[49] = ultralisk_count
        current_state[50] = baneling_count
        current_state[51] = infestor_count
        current_state[52] = swarmhost_count
        current_state[53] = ravager_count
        current_state[54] = viper_count
        current_state[55] = creep_tumor_count
        current_state[56] = self.counter2
		
        hot_squares = np.zeros(4)
        enemy_y, enemy_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] +1)/32))
            x = int(math.ceil((enemy_x[i] +1)/32))
			
            hot_squares[((y-1) *2) + (x - 1)] = 1
        if not self.base_top_left:
            hot_squares = hot_squares[::-1]
			
        for i in range(0,4):
            current_state[i+56] = hot_squares[i]
        green_squares = np.zeros(4)
        friendly_y, friendly_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        for i in range(0, len(friendly_y)):
            y=int(math.ceil((friendly_y[i] +1)/32))
            x=int(math.ceil((friendly_x[i] +1)/32))
            green_squares[((y-1)*2)+(x-1)]=1
			
        if not self.base_top_left:
            green_squares = green_squares[::-1]
        for i in range (0,4):
            current_state[i+60] = green_squares[i]
        if self.previous_action is not None:
            self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))
        
        rl_action = self.qlearn.choose_action(str(current_state))
        self.previous_state = current_state
        self.previous_action = rl_action
		
        smart_action, x,y = self.splitAction(self.previous_action)
		
        if smart_action == ACTION_RESEARCH_CENTRIFUGAL_HOOKS and self.can_do(obs,ACTION_RESEARCH_CENTRIFUGAL_HOOKS):
            unit_y,unit_x = (unit_type == BANELINGNEST).nonzero()
            if unit_y.any():
                i=random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails= fails +1
        elif smart_action == ACTION_RESEARCH_MELEE_WEAPONS or smart_action == ACTION_RESEARCH_GROUND_ARMOR or smart_action == ACTION_RESEARCH_MISSLE_WEAPONS:
            unit_y,unit_x = (unit_type == EVOLUTIONCHAMBER).nonzero()
            if unit_y.any():
                i=random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails= fails +1
        elif smart_action == ACTION_BUILD_BROODLORD:
            unit_y,unit_x = (unit_type == CORRUPTOR).nonzero()
            
            if unit_y.any():
                i=random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails= fails +1
        elif smart_action == ACTION_RESEARCH_CHITINOUS_PLATING:
            unit_y,unit_x = (unit_type == ULTRALISKCAVERN).nonzero()
            if unit_y.any():
                i=random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_RESEARCH_NEURAL_PARASITE or smart_action == ACTION_RESEARCH_PATHOGEN_GLANDS:
            unit_y,unit_x = (unit_type == INFESTATIONPIT).nonzero()
            if unit_y.any():
                i=random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails = fails+1
        
        elif smart_action == ACTION_RESEARCH_ADAPTIVE_TALONS:
            unit_y,unit_x = (unit_type == LURKERDEN).nonzero()
            if unit_y.any():
                i=random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_RESEARCH_MUSCULAR_AUGMENTS or smart_action == ACTION_RESEARCH_GROOVED_SPINES:
            unit_y,unit_x = (unit_type == HYDRALISKDEN).nonzero()
            if unit_y.any():
                i=random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_RESEARCH_GLIAL_RECONSTITUTION or smart_action == ACTION_RESEARCH_TUNNELING_CLAWS:
            unit_y,unit_x = (unit_type == ROACHWARREN).nonzero()
            if unit_y.any():
                i=random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_RESEARCH_ADRENAL_GLANDS or smart_action == ACTION_RESEARCH_METABOLIC_BOOST:
            unit_y,unit_x = (unit_type == SPAWN).nonzero()
            if unit_y.any():
                i=random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails = fails+1
            
        elif smart_action == ACTION_MORPH_OVERSIGHT or smart_action == ACTION_CHANGELING:
            unit_y, unit_x = (unit_type == OVERSEER).nonzero()
            if unit_y.any():
                i = random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails = fails+1
            
        elif smart_action == ACTION_CREEP_TUMOR:
            unit_y, unit_x = (unit_type == CREEPTUMOR).nonzero()
            if unit_y.any():
                i = random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_BUILD_RAVAGER and self.can_do(obs,_MORPH_RAVAGER):
            unit_y, unit_x = (unit_type == ROACH).nonzero()
            if unit_y.any():
                i = random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT,[_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_RESEARCH_BURROW or smart_action == ACTION_RESEARCH_PNEUMATIZED_CARAPACE:
            unit_y, unit_x = (unit_type == LAIR).nonzero()
            if unit_y.any():
                i = random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT,[_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_MORPH_LURKERDEN:
            unit_y, unit_x = (unit_type == HYDRALISKDEN).nonzero()
            if unit_y.any():
                i = random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT,[_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_MORPH_LURKER:
            unit_y, unit_x = (unit_type == HYDRALISK).nonzero()
            if unit_y.any():
                i = random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT,[_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_BUILD_GREATERSPIRE or smart_action == ACTION_RESEARCH_FLYING_ARMOR or smart_action == ACTION_RESEARCH_FLYING_ATTACK:
            unit_y, unit_x = (unit_type == SPIRE).nonzero()
            if unit_y.any():
                i = random.randint(0,len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT,[_NOT_QUEUED,target])
        elif smart_action == ACTION_MORPH_OVERSEER:
            unit_y, unit_x =( unit_type == OVERLORD).nonzero()
            if unit_y.any():
                i = random.randint(0, len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_BUILD_BANELING:
            unit_y, unit_x =( unit_type == ZERGLING).nonzero()
            if unit_y.any():
                i = random.randint(0, len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_BUILD_CREEP_TUMOR or smart_action == ACTION_MAKE_LARVA or smart_action == ACTION_TRANSFUSION:
            if q_y.any():
                unit_y, unit_x = (unit_type == QUEEN).nonzero()
                i = random.randint(0, len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT,[_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_BUILD_QUEEN or smart_action == ACTION_MORPH_HIVE or smart_action == ACTION_MORPH_LAIR or smart_action == ACTION_RALLY_WORKERS or smart_action == ACTION_RESEARCH_BURROW:
            if h_y.any():
               unit_y,unit_x = (unit_type == HATCHERY).nonzero()
               i = random.randint(0, len(unit_y)-1)
               target = [unit_x[i],unit_y[i]]
               succeeds = succeeds +1
               return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_BUILD_SPIRE or smart_action == ACTION_BUILD_LURKERDEN or smart_action == ACTION_BUILD_SPORECRAWLER or smart_action == ACTION_BUILD_SPINECRAWLER or smart_action == ACTION_BUILD_ROACHWARREN or smart_action == ACTION_BUILD_INFESTATIONPIT or smart_action == ACTION_BUILD_ULTRALISKCAVERN or smart_action == ACTION_BUILD_HATCHERY or smart_action == ACTION_BUILD_SPAWN or smart_action == ACTION_BUILD_EXTRACTOR or smart_action == ACTION_BUILD_BANELING_NEST or smart_action == _BUILD_HYDRALISKDEN or smart_action == _BUILD_EVOCHAMBER:
            unit_y,unit_x = (unit_type == DRONE).nonzero()
			
            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_MINE_GAS or smart_action == ACTION_HARVEST_MINERAL:
            if self.can_do(obs,_SELECT_IDLE_WORKER):
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_IDLE_WORKER, [_NOT_QUEUED])
            else:
                fails = fails+1
            
        elif smart_action == ACTION_BUILD_CORRUPTER or smart_action == ACTION_BUILD_VIPER or  smart_action == ACTION_BUILD_MUTALISK or smart_action == ACTION_BUILD_SWARMHOST or smart_action == ACTION_BUILD_ROACH or smart_action == ACTION_BUILD_INFESTOR or smart_action == ACTION_BUILD_ULTRALISK or smart_action == ACTION_BUILD_ZERGLING or smart_action == ACTION_BUILD_OVERLORD or smart_action == ACTION_BUILD_DRONE or smart_action == ACTION_BUILD_HYDRALISK:
            if l_y.any():
                i = random.randint(0, len(l_y) -1)
                target = [l_x[i], l_y[i]]
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_ATTACK or smart_action == ACTION_MOVE or smart_action == ACTION_BURROW or smart_action == ACTION_BURROW_UP:
            if _SELECT_ARMY in obs.observation['available_actions']:
                succeeds = succeeds +1
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            else:
                fails = fails+1
    elif self.move_number ==1:
        self.move_number  +=1
        smart_action, x,y = self.splitAction(self.previous_action)
        
        if smart_action == ACTION_BUILD_EXTRACTOR and self.can_do(obs,_BUILD_EXTRACTOR):
            unit_y, unit_x = (unit_type == _VESPENE_GEYSER).nonzero()
            if unit_y.any():
                i= random.randint(0, len(unit_y) -1)
                x = unit_x[i]
                y = unit_y[i]
                target= [int(x),int(y)]
                succeeds = succeeds +1
                return actions.FunctionCall(_BUILD_EXTRACTOR, [_NOT_QUEUED,target])
            else:
                fails = fails+1
        elif smart_action == ACTION_BUILD_CREEP_TUMOR and self.can_do(obs,_BUILD_CREEPTUMOR):
            target= self.findLocationForBuilding(_BUILD_CREEPTUMOR)
            if target == 0:
                fails = fails+1
                return actions.FunctionCall(_NO_OP, [])
            succeeds = succeeds +1
            return actions.FunctionCall(_BUILD_CREEPTUMOR, [_NOT_QUEUED,target])
        elif smart_action == ACTION_MORPH_OVERSEER and self.can_do(obs,_MORPH_OVERSEER):
            succeeds = succeeds +1			
            return actions.FunctionCall(_MORPH_OVERSEER,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_ROACH and self.can_do(obs,_TRAIN_ROACH):	
            succeeds = succeeds +1					
            return actions.FunctionCall(_TRAIN_ROACH,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_QUEEN and self.can_do(obs,_BUILD_QUEEN):	
            succeeds = succeeds +1					
            return actions.FunctionCall(_BUILD_QUEEN,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_SPAWN and self.can_do(obs,_BUILD_SPAWN):
            target= self.findLocationForBuilding(_BUILD_SPAWN)
            if target == 0:
                fails = fails+1
                return actions.FunctionCall(_NO_OP, [])
            succeeds = succeeds +1			
            return actions.FunctionCall(_BUILD_SPAWN, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_BANELING_NEST and self.can_do(obs,_BUILD_BANELINGNEST):
            target= self.findLocationForBuilding(_BUILD_BANELINGNEST)
            if target == 0:
                fails = fails+1
                return actions.FunctionCall(_NO_OP, [])
            return actions.FunctionCall(_BUILD_BANELINGNEST, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_EVOCHAMBER and self.can_do(obs,_BUILD_EVOCHAMBER):
            target= self.findLocationForBuilding(_BUILD_EVOCHAMBER)
            if target == 0:
                fails = fails+1
                return actions.FunctionCall(_NO_OP, [])
            return actions.FunctionCall(_BUILD_EVOCHAMBER, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_ULTRALISKCAVERN and self.can_do(obs,_BUILD_ULTRALISKCAVERN):
            target= self.findLocationForBuilding(_BUILD_ULTRALISKCAVERN)
            if target == 0:
                fails = fails+1
                return actions.FunctionCall(_NO_OP, [])
            succeeds = succeeds +1			
            return actions.FunctionCall(_BUILD_ULTRALISKCAVERN, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_ROACHWARREN and self.can_do(obs,_BUILD_ROACHWARREN):
            target= self.findLocationForBuilding(_BUILD_ROACHWARREN)
            if target == 0:
                fails = fails+1
                return actions.FunctionCall(_NO_OP, [])
            return actions.FunctionCall(_BUILD_ROACHWARREN, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_SPORECRAWLER and self.can_do(obs,_BUILD_SPORECRAWLER):
            target= self.findLocationForBuilding(_BUILD_SPORECRAWLER)
            if target == 0:
                fails = fails+1
                return actions.FunctionCall(_NO_OP, [])
            return actions.FunctionCall(_BUILD_SPORECRAWLER, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_SPINECRAWLER and self.can_do(obs,_BUILD_SPINECRAWLER):
            target= self.findLocationForBuilding(_BUILD_SPINECRAWLER)
            if target == 0:
                fails = fails+1
                return actions.FunctionCall(_NO_OP, [])
            return actions.FunctionCall(_BUILD_SPINECRAWLER, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_INFESTATIONPIT and self.can_do(obs,_BUILD_INFESTATIONPIT):
            target= self.findLocationForBuilding(_BUILD_INFESTATIONPIT)
            if target == 0:
                 return actions.FunctionCall(_NO_OP, [])
            return actions.FunctionCall(_BUILD_INFESTATIONPIT, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_NYDUSNETWORK and self.can_do(obs,_BUILD_NYDUSNETWORK):
            target= self.findLocationForBuilding(_BUILD_NYDUSNETWORK)
            if target == 0:
                fails = fails+1
                return actions.FunctionCall(_NO_OP, [])
            succeeds = succeeds +1			
            return actions.FunctionCall(_BUILD_NYDUSNETWORK, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_HYDRALISKDEN and self.can_do(obs,_BUILD_HYDRALISKDEN):
            target= self.findLocationForBuilding(_BUILD_HYDRALISKDEN)
            if target == 0:
                return actions.FunctionCall(_NO_OP, [])
            return actions.FunctionCall(_BUILD_HYDRALISKDEN, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_HATCHERY and self.can_do(obs,_BUILD_HATCHERY):
            target= self.findLocationForBuilding(_BUILD_HATCHERY)
            if target == 0:
                return actions.FunctionCall(_NO_OP, [])
            return actions.FunctionCall(_BUILD_HATCHERY, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_NYDUSWORK and self.can_do(obs,_BUILD_NYDUSWORK):
            target= self.findLocationForBuilding(_BUILD_NYDUSCANAL)
            if target == 0:
                fails = fails+1
                return actions.FunctionCall(_NO_OP, [])
            succeeds = succeeds +1			
            return actions.FunctionCall(_BUILD_NYDUSWORK, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_SPIRE and self.can_do(obs,_BUILD_SPIRE):
            target= self.findLocationForBuilding(_BUILD_SPIRE)
            if target == 0:
                fails = fails+1
                return actions.FunctionCall(_NO_OP, [])
            succeeds = succeeds +1			
            return actions.FunctionCall(_BUILD_SPIRE, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_LURKERDEN and self.can_do(obs,_BUILD_LURKERDEN):
            succeeds = succeeds +1			
            return actions.FunctionCall(_BUILD_LURKERDEN, [_NOT_QUEUED,target])
        elif smart_action == ACTION_BUILD_VIPER and self.can_do(obs,_TRAIN_VIPER):
            succeeds = succeeds +1			
            return actions.FunctionCall(_TRAIN_VIPER,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_ADRENAL_GLANDS and self.can_do(obs,_RESEARCH_ADRENAL_GLANDS):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_ADRENAL_GLANDS,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_CHITINOUS_PLATING and self.can_do(obs,_RESEARCH_CHITINOUS_PLATING):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_CHITINOUS_PLATING,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_METABOLIC_BOOST and self.can_do(obs,_RESEARCH_METABOLIC_BOOST):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_METABOLIC_BOOST,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_GROUND_ARMOR and self.can_do(obs,_RESEARCH_ZERG_GROUND_ARMOR):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_ZERG_GROUND_ARMOR,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_FLYING_ARMOR and self.can_do(obs,_RESEARCH_ZERG_FLYER_ARMOR):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_ZERG_FLYER_ARMOR,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_FLYING_ATTACK and self.can_do(obs,_RESEARCH_ZERG_FLYER_ATTACK):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_ZERG_FLYER_ATTACK,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_MISSLE_WEAPONS and self.can_do(obs,_RESEARCH_ZERG_MISSILE_WEAPONS):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_ZERG_MISSILE_WEAPONS,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_MELEE_WEAPONS and self.can_do(obs,_RESEARCH_ZERG_MELEE_WEAPONS):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_ZERG_MELEE_WEAPONS,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_TUNNELING_CLAWS and self.can_do(obs,_RESEARCH_TUNNELCLAWS):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_TUNNELCLAWS,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_MUSCULAR_AUGMENTS and self.can_do(obs,_RESEARCH_MUSCULAR_AUGMENTS):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_MUSCULAR_AUGMENTS,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_PNEUMATIZED_CARAPACE and self.can_do(obs,_RESEARCH_PNEUMATIZED_CARAPACE):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_PNEUMATIZED_CARAPACE,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_GLIAL_RECONSTITUTION and self.can_do(obs,_RESEARCH_GLIAL_RECONSTITUTION):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_GLIAL_RECONSTITUTION,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_ADAPTIVE_TALONS and self.can_do(obs,_RESEARCH_TALONS):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_TALONS,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_GROOVED_SPINES and self.can_do(obs,_RESEARCH_SPINES):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_SPINES,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_PATHOGEN_GLANDS and self.can_do(obs,_RESEARCH_GLANDS):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_GLANDS,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_CENTRIFUGAL_HOOKS and self.can_do(obs,_RESEARCH_HOOKS):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_HOOKS,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_NEURAL_PARASITE and self.can_do(obs,_RESEARCH_NEURAL_PARASITE):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_NEURAL_PARASITE,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_GROUND_ARMOR and self.can_do(obs,_RESEARCH_ZERG_GROUND_ARMOR):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_ZERG_GROUND_ARMOR,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_GROUND_ARMOR and self.can_do(obs,_RESEARCH_ZERG_GROUND_ARMOR):
            succeeds = succeeds +1			
            return actions.FunctionCall(_RESEARCH_ZERG_GROUND_ARMOR,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_ULTRALISK and self.can_do(obs,_TRAIN_ULTRALISK):
            succeeds = succeeds +1			
            return actions.FunctionCall(_TRAIN_ULTRALISK,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_INFESTOR and self.can_do(obs,_TRAIN_INFESTOR):
            succeeds = succeeds +1			
            return actions.FunctionCall(_TRAIN_INFESTOR,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_ULTRALISK and self.can_do(obs,_TRAIN_ULTRALISK):
            succeeds = succeeds +1			
            return actions.FunctionCall(_TRAIN_ULTRALISK,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_MUTALISK and self.can_do(obs,_TRAIN_MUTALISK):
            succeeds = succeeds +1			
            return actions.FunctionCall(_TRAIN_MUTALISK,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_VIPER and self.can_do(obs,_TRAIN_VIPER):
            succeeds = succeeds +1			
            return actions.FunctionCall(_TRAIN_VIPER,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_CORRUPTER and self.can_do(obs,_TRAIN_CORRUPTER):
            succeeds = succeeds +1			
            return actions.FunctionCall(_TRAIN_CORRUPTER,[_NOT_QUEUED])
        elif smart_action == ACTION_MORPH_LURKER and self.can_do(obs,_MORPH_LURKER):
            succeeds = succeeds +1			
            return actions.FunctionCall(_MORPH_LURKER,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_BROODLORD and self.can_do(obs,_TRAIN_BROODLORD):
            succeeds = succeeds +1			
            return actions.FunctionCall(_TRAIN_BROODLORD,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_SWARMHOST and self.can_do(obs,_TRAIN_SWARMHOST):
            succeeds = succeeds +1			
            return actions.FunctionCall(_TRAIN_SWARMHOST,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_GREATERSPIRE and self.can_do(obs,_MORPH_GREATERSPIRE):
            succeeds = succeeds +1			
            return actions.FunctionCall(_MORPH_GREATERSPIRE,[_NOT_QUEUED])
        elif smart_action == ACTION_MORPH_LAIR and self.can_do(obs,_MORPH_LAIR):
            succeeds = succeeds +1			
            return actions.FunctionCall(_MORPH_LAIR,[_NOT_QUEUED])
        elif smart_action == ACTION_MORPH_HIVE and self.can_do(obs,_MORPH_HIVE):
            succeeds = succeeds +1			
            return actions.FunctionCall(_MORPH_HIVE,[_NOT_QUEUED])
        elif smart_action == ACTION_CHANGELING and self.can_do(obs,_SPAWN_CHANGELING):
            succeeds = succeeds +1			
            return actions.FunctionCall(_ENABLE_CHANGELING,[_NOT_QUEUED])
        
        elif smart_action == ACTION_BUILD_BANELING and self.can_do(obs,_TRAIN_BANELING):
            succeeds = succeeds +1			
            return actions.FunctionCall(_TRAIN_BANELING,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_HYDRALISK and self.can_do(obs,_TRAIN_HYDRALISK):
            succeeds = succeeds +1			
            return actions.FunctionCall(_TRAIN_HYDRALISK,[_NOT_QUEUED])
        elif smart_action == ACTION_BUILD_OVERLORD:
            if self.can_do(obs,_BUILD_OVERLORD):
                succeeds = succeeds +1			
                return actions.FunctionCall(_BUILD_OVERLORD, [_NOT_QUEUED])
            else:
                fails = fails+1
        elif smart_action == ACTION_BUILD_DRONE:
            if self.can_do(obs,_BUILD_DRONE):
                succeeds = succeeds +1			
                return actions.FunctionCall(_BUILD_DRONE, [_NOT_QUEUED])
            else:
                fails = fails+1
        elif smart_action == ACTION_BUILD_ZERGLING:
            if self.can_do(obs,_BUILD_ZERGLING):
                succeeds = succeeds +1			
                return actions.FunctionCall(_BUILD_ZERGLING, [_NOT_QUEUED])
            else:
                fails = fails+1
        elif smart_action == ACTION_MOVE_CAMERA:
            do_it = True
            if do_it and _MOVE_CAMERA in obs.observation["available_actions"]:
                succeeds = succeeds +1			
                return actions.FunctionCall(_MOVE_CAMERA, ([self.transformLocation(int(x),int(y))]))
            else:
                fails = fails+1
        elif smart_action == ACTION_ATTACK:
            do_it = True
            if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                succeeds = succeeds +1			
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x),int(y))])
            else:
                fails = fails+1
                
    elif self.move_number ==2:
        self.move_number =0
        smart_action, x, y = self.splitAction(self.previous_action)
        if smart_action == ACTION_MINE_GAS and e_y.any() and self.can_do(obs,_HARVEST_GATHER):
            unit_y, unit_x = (unit_type == EXTRACTOR).nonzero()
            if unit_y.any():
                i = random.randint(0, len(unit_y) -1)
                x = unit_x[i]
                y = unit_y[i]
                target = [int(x),int(y)]
                succeeds = succeeds +1			
                return actions.FunctionCall(_HARVEST_GATHER, [_NOT_QUEUED,target])
            else:
                fails = fails+1
            return actions.FunctionCall(_HARVEST_GATHER, [_NOT_QUEUED,target])
        elif smart_action == ACTION_MAKE_LARVA and self.can_do(obs,_MAKE_LARVA):
            if h_y.any():
                unit_y,unit_x = (unit_type == HATCHERY).nonzero()
                i = random.randint(0, len(unit_y)-1)
                target = [unit_x[i],unit_y[i]]
                succeeds = succeeds +1			
                try:
                    return actions.FunctionCall(_MAKE_LARVA,([0],target))
                except ValueError:
                    fails = fails + 1
                    pass
            else:
                fails = fails+1
        elif smart_action == ACTION_ATTACK:
            if _ATTACK_MINIMAP in obs.observation['available_actions']:	
                target = [int(x),int(y)]
                succeeds = succeeds +1			
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED,target])
            else:
                fails = fails + 1
        elif smart_action == ACTION_HARVEST_MINERAL:
            if _HARVEST_GATHER in obs.observation['available_actions']:
                unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
				
                if unit_y.any():
                    i = random.randint(0, len(unit_y) -1)
                    x = unit_x[i]
                    y = unit_y[i]
			
                    target = [int(x),int(y)]
                    succeeds = succeeds +1			
                    return actions.FunctionCall(_HARVEST_GATHER, [_NOT_QUEUED,target])
                else:
                    fails = fails+1
            else:
                fails = fails + 1
        elif smart_action == ACTION_CANCEL and self.can_do(obs,_CANCEL): 
            succeeds = succeeds +1	
            return actions.FunctionCall(_CANCEL,[_NOT_QUEUED])
        elif smart_action == ACTION_BURROW and self.can_do(obs,_BURROW): 
            succeeds = succeeds +1	
            return actions.FunctionCall(_BURROW,[_NOT_QUEUED])
        elif smart_action == ACTION_BURROW_UP and self.can_do(obs,_BURROW_UP):
            succeeds = succeeds +1	 
            return actions.FunctionCall(_BURROW_UP,[_NOT_QUEUED])
        elif smart_action == ACTION_RESEARCH_BURROW and self.can_do(obs,_RESEARCH_BURROW):
            succeeds = succeeds +1	 
            return actions.FunctionCall(_RESEARCH_BURROW,[_NOT_QUEUED])
        elif smart_action == ACTION_MOVE_SCREEN and self.can_do(obs,_MOVE_SCREEN):
            player_y,player_x = (obs.observation["feature_minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            i = random.randint(0,len(player_y)-1)
            target = (player_x[i],player_y[i])
            succeeds = succeeds +1	
            return actions.FunctionCall(1,[(target)])
    return actions.FunctionCall(_NO_OP,[])

