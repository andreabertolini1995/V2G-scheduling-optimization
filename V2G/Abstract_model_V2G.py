#!/usr/bin/env python
# coding: utf-8

###################### Pyomo Optimization model ############################################################


#IMPORT PACKAGES
from __future__ import division
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core import Var
from random import *
import pandas as pd
import matplotlib.pylab as plt


#CREATE MODEL
model = AbstractModel()

#DEFINE CONSTANTS
Mk = 0.01          #tariff to impose to the EV owner
ExD = 0.75         #expected value of regulation down 
ExU = 0.25         #expected value of regulation up

number_EVs = 100
final_time = 24;   #end-time of the daily scheduling
num_days = 92;   


#DEFINE PARAMETERS
model.time = Param(within=NonNegativeIntegers)          
model.cars = Param(within=NonNegativeIntegers)          
model.days = Param(within=NonNegativeIntegers)   


#DEFINE ITERATORS
model.T = RangeSet(1, model.time)           
model.I = RangeSet(1, model.cars)  
model.J = RangeSet(1, model.days)   


#DATA STRUCTURES - DECISION VARIABLES
POP_dict = {}
for i in range(1, number_EVs+1): 
    for j in range(1, num_days+1):
        for t in range(1, final_time+1):
            POP_dict[i, j, t] = 0 
            #print(i, j, t, POP_dict[i,j,t])
            
MxAP_dict = {}
for i in range(1, number_EVs+1): 
    for j in range(1, num_days+1):
        for t in range(1, final_time+1):
            MxAP_dict[i, j, t] = 0 
            #print(i, j, t, MxAP_dict[i,j,t])

MnAP_dict = {}
for i in range(1, number_EVs+1): 
    for j in range(1, num_days+1):
        for t in range(1, final_time+1):
            MnAP_dict[i, j, t] = 0 
            #print(i, j, t, MnAP_dict[i,j,t])

Deg_dict = {}
for i in range(1, number_EVs+1): 
    for j in range(1, num_days+1):
        for t in range(1, final_time+1):
            Deg_dict[i, j, t] = 0 
            #print(i, j, t, Deg_dict[i,j,t])
            

#DEFINE DECISION VARIABLES 
model.POP = Var(model.I, model.J, model.T, domain=Reals, initialize = POP_dict)
model.MxAP = Var(model.I, model.J, model.T, domain=NonNegativeReals, initialize = MxAP_dict)
model.MnAP = Var(model.I, model.J, model.T, domain=NonNegativeReals, initialize = MnAP_dict)
model.Deg = Var(model.I, model.J, model.T, domain=NonNegativeReals, initialize = Deg_dict)


#DEFINE PARAMETERS (2)
model.Arrival = Param(model.I, model.J, within=NonNegativeIntegers)
model.T_Trip = Param(model.I, model.J, within=NonNegativeIntegers)

model.EVPer = Param(model.J, model.T, within = NonNegativeReals)           #model the availability of EVs to perform smart charging (or V2G)
model.Dep = Param(model.I, model.J, model.T, within = NonNegativeReals)    #probability of unexpected departure
model.Comp = Param(model.I, model.J, model.T, within = NonNegativeReals)   

model.SOC = Param(model.I, model.J, within = NonNegativeReals)             #initial state of charge of the cars 


#Technical specifications of the selected car and charger
model.Ef = Param(model.I, within = NonNegativeReals)
model.Mc = Param(model.I, within = NonNegativeReals)  

model.DC = Param(model.I, within = NonNegativeReals)

model.MP = Param(model.I, model.J, model.T) 


#Prices (electricity + ancilarry services)
model.Price_RD = Param(model.J, model.T, within = NonNegativeReals)
model.Price_RU = Param(model.J, model.T, within = NonNegativeReals)
model.P = Param(model.J, model.T, within = NonNegativeReals)   


#OBJECTIVE FUNCTION
def income(model):
    return sum(sum((model.Price_RD[j,t] * sum(model.MxAP[i,j,t] for i in model.I) + model.Price_RU[j,t] * sum(model.MnAP[i,j,t] for i in model.I)) * model.EVPer[j,t] for t in model.T) for j in model.J) + Mk*sum(sum(sum((model.MxAP[i,j,t] * ExD + model.POP[i,j,t] - model.MnAP[i,j,t] * ExU) * model.EVPer[j,t] for t in model.T) for j in model.J) for i in model.I)
        
def costs(model):
    return sum(sum(sum((model.MxAP[i,j,t] * ExD + model.POP[i,j,t] - model.MnAP[i,j,t] * ExU) * model.EVPer[j,t] * model.P[j,t] for t in model.T) for j in model.J) for i in model.I) + sum(sum(sum(model.Deg[i,j,t] for t in model.T) for j in model.J) for i in model.I)

#Final equation
def profit(model):
    return income(model) - costs(model)
model.obj = Objective(rule = profit, sense = maximize)


#OPERATIONAL CONSTRAINTS
def fifteen_rule(model, i):            #the constraint is applied FOR EVERY EV
    for j in range(1, num_days+1):     #looping through all the days of the simulation
        for k in range(model.Arrival[i,j], model.T_Trip[i,j]):
            return sum((model.MxAP[i,j,t] * ExD + model.POP[i,j,t] - model.MnAP[i,j,t] * ExU) * model.Comp[i,j,t] + (model.Deg[i,j,t]/model.DC[i])*(1-model.Ef[i]*model.Ef[i])/model.Ef[i] for t in range(model.Arrival[i,j], model.T_Trip[i,j])[:k-model.Arrival[i,j]+1]) * model.Ef[i] + model.SOC[i,j] <= model.Mc[i]
model.fifteen = Constraint(model.I, rule=fifteen_rule)


def sixteen_rule(model, i):
    for j in range(1, num_days+1):     
        for k in range(model.Arrival[i,j], model.T_Trip[i,j]):
            return sum((model.MxAP[i,j,t] * ExD + model.POP[i,j,t] - model.MnAP[i,j,t] * ExU) * model.Comp[i,j,t] + (model.Deg[i,j,t]/model.DC[i])*(1-model.Ef[i]*model.Ef[i])/model.Ef[i] for t in range(model.Arrival[i,j], model.T_Trip[i,j])[:k-model.Arrival[i,j]+1]) * model.Ef[i] + model.SOC[i,j] >= 0
model.sixteen = Constraint(model.I, rule=sixteen_rule)


def seventeen_rule(model, i):          
    for j in range(1, num_days+1):     
        return sum((model.MxAP[i,j,t] * ExD + model.POP[i,j,t] - model.MnAP[i,j,t] * ExU) * model.Comp[i,j,t] + (model.Deg[i,j,t]/model.DC[i])*(1-model.Ef[i]*model.Ef[i])/model.Ef[i] for t in range(model.Arrival[i,j], model.T_Trip[i,j])) + model.SOC[i,j] >= 0.99 * model.Mc[i]    
model.seventeen = Constraint(model.I, rule=seventeen_rule)


def eighteen_rule(model, i):
    for j in range(1, num_days+1):     
        return (model.MxAP[i,j,model.Arrival[i,j]] + model.POP[i,j,model.Arrival[i,j]]) * model.Comp[i,j,model.Arrival[i,j]] * model.Ef[i] + model.SOC[i,j] <= model.Mc[i]      
model.eighteen = Constraint(model.I, rule=eighteen_rule)


def nineteen_rule(model, i):
    for j in range(1, num_days+1):     #looping through all the days of the simulation
        return (model.POP[i,j,model.Arrival[i,j]] - model.MnAP[i,j,model.Arrival[i,j]]) + (model.Deg[i,j,model.Arrival[i,j]]/model.DC[i])*(1-model.Ef[i]*model.Ef[i])/model.Ef[i] * model.Comp[i,j,model.Arrival[i,j]] * model.Ef[i] + model.SOC[i,j] >= 0      
model.nineteen = Constraint(model.I, rule=nineteen_rule)


#Equation 20 is about the Trip constant that I am not considering right now (I will soon)


def twenty_one_rule(model, i, j, t):
    return (model.MxAP[i,j,t] + model.POP[i,j,t]) * model.Comp[i,j,t] <= model.MP[i,j,t] 
model.twenty_one = Constraint(model.I, model.J, model.T, rule=twenty_one_rule)


def twenty_two_rule(model, i, j, t):
    return model.MnAP[i,j,t] <= model.POP[i,j,t] + model.MP[i,j,t] 
model.twenty_two = Constraint(model.I, model.J, model.T, rule=twenty_two_rule)


#equation 23 is about responsive reserve


def twenty_four_rule(model, i, j, t):
    return model.MxAP[i,j,t] >= 0
model.twenty_four = Constraint(model.I, model.J, model.T, rule=twenty_four_rule)


def twenty_five_rule(model, i, j, t):
    return model.MnAP[i,j,t] >= 0
model.twenty_five = Constraint(model.I, model.J, model.T, rule=twenty_five_rule)


#equation 26 is about responsive reserve


def twenty_seven_rule(model, i, j, t):
    return model.POP[i,j,t] >= -model.MP[i,j,t]
model.twenty_seven = Constraint(model.I, model.J, model.T, rule=twenty_seven_rule)


def twenty_eight_rule(model, i, j, t):
    return model.Deg[i,j,t] >= 0
model.twenty_eight = Constraint(model.I, model.J, model.T, rule=twenty_eight_rule)


def twenty_nine_rule(model, i, j, t):
    return model.Deg[i,j,t] >= model.DC[i] * (model.POP[i,j,t] - model.MnAP[i,j,t] * ExU) * model.Comp[i,j,t]/model.Ef[i]
model.twenty_nine = Constraint(model.I, model.J, model.T, rule=twenty_nine_rule)


#equation 30 is about the forecasted peak load 


#SOLVE THE PROBLEM
opt = SolverFactory('glpk')

instance = model.create_instance('abstract_model_V2G.dat') #import of the dat file
results = opt.solve(instance, tee=True)
results.write()
instance.solutions.load_from(results)


POP_solutions_dict = {}
MxAP_solutions_dict = {}
MnAP_solutions_dict = {}
Deg_solutions_dict = {}


#Saving the results in separate dictionaries
for v in instance.component_objects(Var, active=True):  #loop thorugh the variables
    #print ("Variable",v)
    varobject = getattr(instance, str(v))
    varobject_string = str(varobject)
    if (varobject_string == "POP"):
        for index in varobject:
            POP_solutions_dict[index] = varobject[index].value   
            #print (index, varobject[index].value)
    elif (varobject_string == "MxAP"):
        for index in varobject:
            MxAP_solutions_dict[index] = varobject[index].value  
            #print (index, varobject[index].value)
    elif (varobject_string == "MnAP"):
        for index in varobject:
            MnAP_solutions_dict[index] = varobject[index].value  
            #print (index, varobject[index].value)
    elif (varobject_string == "Deg"):
        for index in varobject:
            Deg_solutions_dict[index] = varobject[index].value  
            #print (index, varobject[index].value)
    
    


###################### Plot the results ############################################################
# Modify the variable 'car' value in order to get the daily POP, MxAP and MnAP trends for the day considered.
# This is the plot for the whole fleet

day = 23;

POP_plot_fleet = []
RD_plot = []
RU_plot = []


for k in range(1, final_time+1):
    POP_plot_fleet.append(sum(POP_solutions_dict[car, day, k] for car in range(1, number_EVs+1)))
    POP_plot_fleet.append(sum(POP_solutions_dict[car, day, k] for car in range(1, number_EVs+1)))
POP_plot_fleet.pop()
POP_plot_fleet.pop()
    
for k in range(1, final_time+1):
    RD_plot.append(sum(MxAP_solutions_dict[car, day, k] for car in range(1, number_EVs+1)))
    RD_plot.append(sum(MxAP_solutions_dict[car, day, k] for car in range(1, number_EVs+1)))
RD_plot.pop()
RD_plot.pop()

for k in range(1, final_time+1):
    RU_plot.append(sum(MnAP_solutions_dict[car, day, k] for car in range(1, number_EVs+1)))
    RU_plot.append(sum(MnAP_solutions_dict[car, day, k] for car in range(1, number_EVs+1)))
RU_plot.pop()
RU_plot.pop()
    
#I actually want to plot POP+MxAP and POP-MnAP
for i in range(len(POP_plot_fleet)):
    RD_plot[i] = POP_plot_fleet[i] + RD_plot[i]

for i in range(len(POP_plot_fleet)):
    RU_plot[i] = POP_plot_fleet[i] - RU_plot[i]

    
final_time_plot = [1]
for i in range(2, final_time+1):
    final_time_plot.append(i)
    final_time_plot.append(i)
final_time_plot.pop()
    
    
plt.plot(final_time_plot, POP_plot_fleet, label = "POP", c = "Black")
plt.plot(final_time_plot, RD_plot, label = "POP + RD", c = "Red", linestyle = '--')
plt.plot(final_time_plot, RU_plot, label = "POP - RU", c = "Green", linestyle = '--')
plt.xticks(np.arange(1, 26, 3))
plt.yticks(np.arange(-740, 740, 210))  
plt.xlabel("Time")
plt.ylabel("kW")
plt.legend()




