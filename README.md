# Designing Two-Echelon Last-Mile Delivery Networks with Multiple Capacity Levels under Uncertainty

### Authors
- Pitehr Hurtado-Cayo
- Juan C. Pina-Pardo,
- Selene Silvestri,
- Matthias Winkenbach,

## Configuration Parameters
Instance parameters we have:

1. `instance_id` - The ID of the instance.
2. `N` - The number of scenarios to be read for each sample.
3. `M` - The number of samples to be read for testing solution quality.
4. `T` - The number of time periods.
5. `Q` - The number of capacity levels considered in each facilities
6. `type_of_flexibility` - The type of flexibility to be considered.
   1. `FIXED_CAPACITY` - The flexibility is considered in the installation of facilities, i.e., if facilities are installed so that they have to operate always with the same capacity.
   2. `FLEX_CAPACITY` - The flexibility is considered in the installation and operation of facilities, i.e., if facilities are installed so that they can operate with different capacities.
7. `periods` - The periods to be considered.
8.  `facilities` - The satellites to be installed.
    1. `cost installation` - The cost of installing the satellite.
    2. `cost operation` - The cost of operating the satellite.
9.  `scenarios` - The scenarios obtained from sampling the demand.
    1.  `pixel` - The pixel of the scenario.
    2.  `costs` - The costs of serving from DCs and facilities.
    3.  `fleet size` - The fleet size required to serve the demand.

## Formulation
WIP
