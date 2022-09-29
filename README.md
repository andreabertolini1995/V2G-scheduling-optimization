# energy_systems_optimization
Profit optimisation of a V2G energy aggregator participating in the ancillary services and energy market formulated as a Convex Linear Problem using Pyomo.  
## Input data
The input data needed to be first constructed following a certain syntax in a DAT file and, at a later time, fed into the model. In order to make the process more efficient, I created a second Python script in charge of automatically generating the file with all the input data, given the number of vehicles and the number of days considered for the simulation.

### Driving profiles
In my own implementation of the model, I considered a hypothetical group of 100 EVs whose owners arrive at work every morning between 6AM and 9AM and come back home every evening between 4PM and 7PM. During this time window, that dynamically changes every day for every vehicle, the EVs are connected to the grid and, consequently, able to provide ancillary services and exchange energy with it. The initial SOC of the EVs every morning is randomly generated, since during the previous night it’s unknown if the owner travelled or not (and, in case, how far he/she went).

### EVs' and chargers' specifications
I decided to consider that everybody owns a Nissan Leaf. This choice made my life easier when looking for maximum capacity of the car’s battery that, in case of the Leaf, is 24 kWh. The EV charging and discharging efficiencies are assumed to be 90%, while the replacement cost of the battery is considered to be 200€/kWh. For the charger, the maximum possible power draw has been fixed to 7.2 kW.

### Price signals
Due to a greater ease of accessing the data, the Italian electricity market has been taken into account. However, its structure is not easy, especially due to the numerous sub-phases of which the ancillary services market and the balancing market consist, created for adjustments of the price over time. Therefore, the aggregator’s bidding strategy as well as its active participation in the auction process have not been modeled, since it would have constituted an optimization on its own and it would have taken too many resources, in addition to be out of the scope of the project.

Instead, I had considered the average price at which the Transmission System Operator (TSO) pays the services of regulation down and regulation up. I could find this information in the archives of [Terna S.p.A](https://www.terna.it/it/sistema-elettrico/mercato-servizi-dispacciamento), the only Italian TSO. When the price is equal to zero, it means that no quantity of energy has been accepted by Terna that hour. Italy is divided into 6 different bidding areas, each of which with its own prices and related data. I selected the NORD region, believing that the electric grid might be exposed to more problems and because of the higher quantity of data available.

The electricity prices are the same for the whole country and have been taken from the [GME repository](https://www.mercatoelettrico.org/En/Default.aspx) while the aggregator rate of energy charged to the consumer has fixed at $0.01/kWh.

Both ancillary services prices and energy prices span over a simulation period of three months, from August 2019 to October 2019.



