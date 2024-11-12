from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from torch.fx.experimental.unification import variables

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
        ("keyPresent", "Starts")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)

cpd_keypresent = TabularCPD(
    variable="keyPresent", variable_card=2,
    values=[[0.70], [0.30]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"keyPresent": ['yes', "no"],
                 "Starts": ['yes', 'no'] },
)

# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)

car_infer = VariableElimination(car_model)

def question_four():
    # print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))
    print("Given that the car will not move, what is the probability that the battery is not working?")
    print(car_infer.query(variables=["Battery"],evidence={"Moves":"no"}))
    print()

    print("Given that the radio is not working, what is the probability that the car will not start?")
    print(car_infer.query(variables=["Starts"],evidence={"Radio":"Doesn't turn on"}))
    print()

    print("Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?")
    print()

    print("Unknown Gas")
    print(car_infer.query(variables=["Radio"],evidence={"Battery":"Works"}))
    print()

    print("Gas Full")
    print(car_infer.query(variables=["Radio"],evidence={"Battery":"Works", "Gas":"Full"}))
    print()

    print("Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car dies not have gas in it?")
    print()

    print("If we do not know if the car has gas in it")
    print(car_infer.query(variables=["Ignition"],evidence={"Moves":"no"}))
    print()

    print("If we do know that the car has no gas in it")
    print(car_infer.query(variables=["Ignition"],evidence={"Moves":"no", "Gas":"Empty"}))
    print()

    print("What is the probability that the car starts if the radio works and it has gas in it?")
    print()
    print(car_infer.query(variables=["Starts"],evidence={"Radio":"turns on","Gas":"Full"}))



