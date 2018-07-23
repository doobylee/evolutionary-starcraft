"""

Visualize Genetic Algorithm to find a maximum point in a function.



Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import math
import os
import os.path
import numpy as np
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from absl import logging
from pysc2 import maps
from pysc2 import run_configs
from pysc2.env import sc2_env
from pysc2.lib import remote_controller
from pysc2.lib import run_parallel
from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.agents import dooby
from absl import app

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
attacks= []
camera_moves = []
moves = []
counter = 0
		   
		   
running = True
smart_actions = [
 ACTION_MOVE_SCREEN,
 ACTION_CREEP_TUMOR,
 ACTION_BUILD_CREEP_TUMOR,
 ACTION_BUILD_RAVAGER,
 ACTION_RESEARCH_CHITINOUS_PLATING,
 ACTION_RESEARCH_NEURAL_PARASITE,
 ACTION_RESEARCH_PATHOGEN_GLANDS,
 ACTION_RESEARCH_MUSCULAR_AUGMENTS,
 ACTION_RESEARCH_ADAPTIVE_TALONS,
 ACTION_RESEARCH_GROOVED_SPINES,
 ACTION_RESEARCH_GLIAL_RECONSTITUTION,
 ACTION_RESEARCH_TUNNELING_CLAWS,
 ACTION_RESEARCH_CENTRIFUGAL_HOOKS,
 ACTION_BUILD_VIPER,
 ACTION_RESEARCH_METABOLIC_BOOST,
 ACTION_RESEARCH_ADRENAL_GLANDS,
 ACTION_RESEARCH_PNEUMATIZED_CARAPACE,
 ACTION_RESEARCH_MELEE_WEAPONS,
 ACTION_RESEARCH_GROUND_ARMOR,
 ACTION_RESEARCH_FLYING_ARMOR,
 ACTION_RESEARCH_FLYING_ATTACK,
 ACTION_RESEARCH_MISSLE_WEAPONS,
 ACTION_RESEARCH_BURROW,
 ACTION_MORPH_LURKERDEN,
 ACTION_BUILD_SPIRE,
 ACTION_MORPH_LURKER,
 ACTION_MOVE_SCREEN,
 ACTION_BUILD_SWARMHOST,
 ACTION_BUILD_MUTALISK,
 ACTION_BUILD_CORRUPTER,
 ACTION_BUILD_BROODLORD,
 ACTION_BUILD_ROACHWARREN,
 ACTION_BUILD_ROACH,
 ACTION_MORPH_OVERSEER,
 ACTION_BUILD_SPORECRAWLER,
 ACTION_BUILD_SPINECRAWLER,
 ACTION_DO_NOTHING,
 ACTION_BUILD_OVERLORD,
 ACTION_BUILD_ZERGLING,
 ACTION_BUILD_OVERLORD,
 ACTION_BUILD_ZERGLING,
 ACTION_BUILD_SPAWN,
 ACTION_BUILD_DRONE,
 ACTION_BUILD_HATCHERY,
 ACTION_MOVE_CAMERA,
 ACTION_HARVEST_MINERAL,
 ACTION_BUILD_QUEEN,
 ACTION_MAKE_LARVA,
 ACTION_BUILD_EXTRACTOR,
 ACTION_MOVE_SCREEN,
 ACTION_MINE_GAS,
 ACTION_BUILD_BANELING_NEST,
 ACTION_BUILD_HYDRALISKDEN,
 ACTION_BUILD_EVOCHAMBER,
 ACTION_BUILD_BANELING,
 ACTION_BUILD_HYDRALISK,
 ACTION_MORPH_LAIR,
 ACTION_MORPH_HIVE,
 ACTION_BUILD_ULTRALISKCAVERN,
 ACTION_BUILD_ULTRALISK,
 ACTION_BUILD_INFESTATIONPIT,
 ACTION_BUILD_INFESTOR,
 ACTION_BUILD_GREATERSPIRE 
] 
		 	
counter = 0
for mm_x in range(0, 64):

   for mm_y in range(0, 64):

       if (mm_x+1) % 16 == 0 and (mm_y+1) % 16 == 0 and mm_x != 64 and mm_y != 64:

           attacks.append((ACTION_ATTACK + '_' + str(mm_x) + '_' + str(mm_y)))
           camera_moves.append((ACTION_MOVE_CAMERA + '_' + str(mm_x) + '_' + str(mm_y)))
           moves.append((ACTION_MOVE_SCREEN + '_' + str(mm_x) + '_' + str(mm_y)))
           smart_actions.append(moves[counter])
           smart_actions.append(attacks[counter])
           smart_actions.append(camera_moves[counter]) 
           counter = counter +1
		 	
p = 0
DNA_SIZE = len(smart_actions)*100
POP_SIZE = 5
CROSS_RATE = 0.5         # mating probability (DNA crossover)

MUTATION_RATE = 0.003    # mutation probability
GENERATION = 0
N_GENERATIONS = 100





pop = []





def generateDNA():
    template=[]
    for x in range(0,POP_SIZE):
        for i in range(0,DNA_SIZE):
            number = random.randint(0,len(smart_actions)-1)
            action = smart_actions[number]
            template.append(action)
        pop.append(template)
if len(pop) <POP_SIZE:
    pop = []
generateDNA()






def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i = random.randint(0, len(parent)-1) 
        n = random.randint(0, len(parent)-1)   # choose crossover points
        number = abs(i-n)
        lower_number = 0
        if i < n :
            lower_number = i
        else:
            lower_number = n
        for i in range(0,number):
            parent[i+lower_number] = pop[i+lower_number]
    return parent





def mutate(child):

    for point in range(0,DNA_SIZE):

        if np.random.rand() < MUTATION_RATE:

            child[point] = smart_actions[random.randint(0,len(smart_actions)-1)]

    return child

def main(unused_argv):
    filenamelist = []
    fitnesslist = []
    fittestofthefit = []
    fittestofthefitfitness = []
    for i in range(0,POP_SIZE):
      fitnesslist.append(0)
    for i in range(N_GENERATIONS):
      GENERATION = i
      print("Most fit DNA: " + str(max(fitnesslist)))
      if GENERATION > 0:
        value = max(fitnesslist)
        for d in range(0,POP_SIZE):
          if value == fitnesslist[d]:
              string2 = str('FittestAgentFile' + str(d) +'_generation' + str(GENERATION-1) + '.gz')
              if os.path.isfile(r'C:\Users\lamdr\Miniconda3\Lib\site-packages\pysc2\agents' + '/' + filenamelist[d] + '.gz'): 
                os.rename((filenamelist[d]+'.gz'),string2)
                fittestofthefit.append(string2)
                fittestofthefitfitness.append(fitnesslist[d])
                filenamelist.remove(filenamelist[d])
            
        for i in range (len(filenamelist)):
            if os.path.isfile(r'C:\Users\lamdr\Miniconda3\Lib\site-packages\pysc2\agents' + '/' + filenamelist[i] + '.gz'): 
                os.remove(filenamelist[i]+'.gz')
        filenamelist = []
        fitnesslist = []
        for i in range(0,POP_SIZE):
            fitnesslist.append(0)
        pop_copy = pop[d].copy()
        parent = pop[d]
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child    
      reward = 0
      i=0
      value = 3 + (1*GENERATION)    
      number = 0
      double = False
      while i<value:
        for s in range(0,POP_SIZE):
          number = 0
          number2 = 0
          for k in range(0,POP_SIZE):
              agent= dooby.ZergAgent()
              agent2= dooby.ZergAgent()
              agent.set_actions(pop[s])
              agent2.set_actions(pop[k]) 
              with sc2_env.SC2Env(map_name="Simple64",players=[sc2_env.Agent(sc2_env.Race.zerg),sc2_env.Agent(sc2_env.Race.zerg)],agent_interface_format=features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=84, minimap=64), use_feature_units=True),step_mul=8, game_steps_per_episode=(0),visualize=True) as env:
                  agent.setup(env.observation_spec(), env.action_spec())
                  agent2.setup(env.observation_spec(), env.action_spec())
                  env._episode_steps   
                  timesteps = env.reset() 
                  agent.reset()
                  agent2.reset()
                  while True:
                    if k == s :
                       string2 = str('AgentFile' + str(k) +'_generation' + str(GENERATION) + '2')
                       filenamelist.append(string2)
                    else:
                       string2 = str('AgentFile' + str(k) +'_generation' + str(GENERATION))
                    string = str('AgentFile' + str(s) +'_generation' + str(GENERATION))
                    filenamelist.append(string)
                    
                    step_actions = [agent.step(timesteps[0],string),agent2.step(timesteps[1],string2)]
         
                    if timesteps[0].last():
                      print('here')
                      print(s)
                      number = number + agent.reward
                      if s == k :
                        number = number + agent2.reward                     
                      else:
                        number2 = number + agent2.reward
                      if s != k:
                        fitnesslist[k] = fitnesslist[k] + (number2/(value*(POP_SIZE*2)))
                      fitnesslist[s] = fitnesslist[s] + (number/ (value*(POP_SIZE*2)))
                      print((number/(value*(POP_SIZE*2))))
                      break
                    timesteps = env.step(step_actions)
        i= i+1
        print(i)
		
		
if __name__ == "__main__":
    app.run(main)

