Energy Conversion and Management - Wind farm layout optimization based on support vector regression guided genetic algorithm with consideration of participation among landowners

# Wind farm layout optimization problem (WFLOP)  SUGGA Python toolbox 

## Reference
[1] Xinglong Ju, Feng Liu, Li Wang, and Wei-Jen Lee. "Wind farm layout optimization based on support vector regression guided genetic algorithm with consideration of participation among landowners." *Energy Conversion and Management* 196 (2019): 1267-1281. DOI: <a href="https://doi.org/10.1016/j.enconman.2019.06.082" target="_blank">10.1016/j.enconman.2019.06.082</a><br/>

If you have any questions, please email Feng Liu (*fliu0@mgh.harvard.edu*), or Xinglong Ju (*xinglong.ju@mavs.uta.edu*).

<p align="center"> 
    <img width="400" src="/IMAGES/Jensen's wake model.png" alt="Jensen's wake model.png"/><br/>
    Jensen's wake model
</p>

## SUGGA - WFLOP Python toolbox guide
### Import necessary libraries
```python
import numpy as np
import pandas as pd
import WindFarmGenetic # wind farm layout optimization using genetic algorithms classes
from datetime import datetime
import os
from sklearn.svm import SVR
import pickle
```

### Wind farm settings and algorithm settings
```python
# parameters for the genetic algorithm
elite_rate = 0.2
cross_rate = 0.6
random_rate = 0.5
mutate_rate = 0.1

wt_N =  25 # number of wind turbines 15, 20, or 25
# NA_loc_array : not available location array, the index starting from 1

# L1 : wind farm, cells 121(inclusive) to 144(inclusive)
NA_loc_array = np.arange(121, 145, 1)
#
# # L2
# NA_loc_array = np.arange(61, 85, 1)
#
# # L3
# NA_loc_array = np.concatenate((np.arange(11, 144, 12), np.arange(12, 145, 12)))
#
# # L4
# NA_loc_array = np.concatenate((np.arange(6, 144, 12), np.arange(7, 145, 12)))
#
# # L5
# NA_loc_array = np.concatenate((np.arange(41, 105, 12), np.arange(42, 105, 12),
#                                np.arange(43, 105, 12),
#                                np.arange(44, 105, 12)))
#
# # L6
# NA_loc_array = np.concatenate((np.arange(1, 28, 12), np.arange(2, 28, 12),
#                                np.arange(12, 37, 12),
#                                np.arange(11, 37, 12),
#                                np.arange(109, 145, 12), np.arange(119, 145, 12),
#                                np.arange(110, 145, 12),
#                                np.arange(120, 145, 12),
#                                ))
#
# # L7
# NA_loc_array = np.arange(133, 145, 1)
#
# # L8
# NA_loc_array = np.arange(61, 73, 1)
#
# # L9
# NA_loc_array = np.arange(12, 145, 12)
#
# # L10
# NA_loc_array = np.arange(6, 145, 12)
#
# # L11
# NA_loc_array = np.concatenate((np.arange(42, 105, 12),
#                                np.arange(43, 105, 12)))
#
# # L12
# NA_loc_array = np.array((1, 2, 11, 12, 13, 24, 121, 132, 133, 134, 143, 144))

# convert numpy array to list, datatype convert
NA_loc = NA_loc_array.tolist()

# L0
# NA_loc = []


population_size = 120  # how many layouts in a population
iteration_times = 200  # how many iterations in a genetic algorithm run

n_inits = 100  # number of initial populations n_inits >= run_times
run_times = 100  # number of different initial populations



# wind farm size, cells
cols_cells = 12  # number of cells each row
rows_cells = 12  # number of cells each column
cell_width = 77.0 * 3 # unit : m

# all data will be save in data folder
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
```


### Create a WindFarmGenetic object
```python
# create an WindFarmGenetic object. Specify the number of rows and the number columns of the wind farm land. N is the number of wind turbines.
# NA_loc is the not available locations on the wind farm land. Landowners does not want to participate in the wind farm.
# pop_size: how many individuals in the population
# iteration: iteration times of the genetic algorithm
wfg = WindFarmGenetic.WindFarmGenetic(rows=cols_cells, cols=rows_cells, N=wt_N, NA_loc=NA_loc, pop_size=population_size,
                                      iteration=iteration_times,cell_width=cell_width, elite_rate=elite_rate,
                                             cross_rate=cross_rate, random_rate=random_rate, mutate_rate=mutate_rate)
# Specify the wind distribution
# wind distribution is discrete (number of wind speeds) by (number of wind directions)
# wfg.init_1_direction_1_N_speed_13()
# file name to store the wind power distribution SVR model
# svr_model_filename = 'svr_1s1d_N_13.svr'

# wfg.init_4_direction_1_speed_13()
# svr_model_filename = 'svr_1s4d_13.svr'

wfg.init_6_direction_1_speed_13()
# svr_model_filename = 'svr_1s6d_13.svr'
```

### Generate initial populations
```python
# initial population saved folder
init_pops_data_folder = "{}/init_data".format(data_folder)
if not os.path.exists(init_pops_data_folder):
    os.makedirs(init_pops_data_folder)
# generate initial populations to start with and store them
# in order to start from the same initial population for different methods
# so it is fair to compare the final results

for i in range(n_inits):
    wfg.gen_init_pop_NA()
    wfg.save_init_pop_NA("{}/init_{}.dat".format(init_pops_data_folder,i), "{}/init_{}_NA.dat".format(init_pops_data_folder,i))
```

### Create results folder
```python
# results folder
# adaptive_best_layouts_N60_9_20190422213718.dat : best layout for AGA of run index 9
# result_CGA_20190422213715.dat : run time and best eta for CGA method
results_data_folder = "data/results"
if not os.path.exists(results_data_folder):
    os.makedirs(results_data_folder)
# if cg,ag,sg folder does not exist, create these folders. Folders to store the running results
# cg: convertional genetic algorithm
# ag: adaptive genetic algorithm
# sg: support vector regression guided genetic algorithm
cg_result_folder = "{}/cg".format(results_data_folder)
if not os.path.exists(cg_result_folder):
    os.makedirs(cg_result_folder)

ag_result_folder = "{}/ag".format(results_data_folder)
if not os.path.exists(ag_result_folder):
    os.makedirs(ag_result_folder)

sg_result_folder = "{}/sg".format(results_data_folder)
if not os.path.exists(sg_result_folder):
    os.makedirs(sg_result_folder)
    
# resul_arr: run_times by 2 , the first column is the run time in seconds for each run and the second column is the conversion efficiency for the run
result_arr = np.zeros((run_times, 2), dtype=np.float32)
```

### Run conventional genetic algorithm (CGA)
```python
# CGA: Conventional genetic algorithm
for i in range(0, run_times):  # run times
    print("run times {} ...".format(i))
    # load initial population
    wfg.load_init_pop_NA("{}/init_{}.dat".format(init_pops_data_folder,i), "{}/init_{}_NA.dat".format(init_pops_data_folder,i))
    # run the conventional genetic algorithm and return run time and conversion efficiency
    run_time, eta = wfg.conventional_genetic_alg(i,result_folder=cg_result_folder)
    result_arr[i, 0] = run_time
    result_arr[i, 1] = eta
time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
# save the run time and etas to a file
filename = "{}/result_conventional_{}.dat".format(cg_result_folder,time_stamp)
np.savetxt(filename, result_arr, fmt='%f', delimiter="  ")
```

### Run adaptive genetic algorithm (AGA)
```python
# AGA: adaptive genetic algorithm
for i in range(0,run_times):  # run times
    print("run times {} ...".format(i))
    wfg.load_init_pop_NA("{}/init_{}.dat".format(init_pops_data_folder,i),"{}/init_{}_NA.dat".format(init_pops_data_folder,i))
    run_time, eta = wfg.adaptive_genetic_alg(i,result_folder=ag_result_folder)
    result_arr[i, 0] = run_time
    result_arr[i, 1] = eta
time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = "{}/result_adaptive_{}.dat".format(ag_result_folder,time_stamp)
np.savetxt(filename, result_arr, fmt='%f', delimiter="  ")
```

### Run support vector regression guided genetic algorithm (SUGGA)
#### Generate wind distribution surface
```python
n_mc_samples = 10000  # svr train data, number of layouts to average

wds_data_folder = "{}/wds".format(data_folder)
if not os.path.exists(wds_data_folder):
    os.makedirs(wds_data_folder)
# mc : monte-carlo

# number of layouts to generate as the training data for regression
# to build the power distribution surface

# mc_layout.dat file stores layouts only with 0s and 1s. 0 means no turbine here. 1 means one turbine here.
# mc_layout_NA.dat file stores layouts with 0s, 1s and 2s. 2 means no turbine and not available for turbine.
# These two files are used to generate wind power distribution.
# Each file has 10000 lines. Each line is layout.
# gen_mc_grid_with_NA_loc function generates these two files.
train_mc_layouts, train_mc_layouts_NA = WindFarmGenetic.LayoutGridMCGenerator.gen_mc_grid_with_NA_loc(rows_cells,
                                                                                                      cols_cells,
                                                                                                      n_mc_samples,
                                                                                                      wt_N, NA_loc,
                                                                                                      "{}/mc_layout.dat".format(wds_data_folder),
                                                                                                      "{}/mc_layout_NA.dat".format(wds_data_folder))


# wfg.init_1_direction_1_N_speed_13()
# file name to store the wind power distribution SVR model
# svr_model_filename = 'svr_1s1d_N_13.svr'

# wfg.init_4_direction_1_speed_13()
# svr_model_filename = 'svr_1s4d_13.svr'

# wfg.init_6_direction_1_speed_13()
svr_model_filename = 'svr_1s6d_13.svr'


# load Monte-Carlo layouts from a text file. 10000 random layouts
layouts = np.genfromtxt("{}/mc_layout.dat".format(wds_data_folder), delimiter="  ", dtype=np.int32)
# generate the location index coordinate and average power output at each location index coordinate
# location index coordinate : in the cells, the cell with index 1 has location index (0,0) and the cell 2 has (1,0)
# store the location index coordinate in x.dat and average power in y.dat
wfg.mc_gen_xy_NA(rows=rows_cells, cols=cols_cells, layouts=layouts, n=n_mc_samples, N=wt_N, xfname="{}/x.dat".format(wds_data_folder),
                 yfname="{}/y.dat".format(wds_data_folder))


# read index location coordinates
x_original = pd.read_csv("{}/x.dat".format(wds_data_folder), header=None, nrows=rows_cells * cols_cells, delim_whitespace=True, dtype=np.float32)
x_original = x_original.values

# read the power output of each index location coordinate
y_original = pd.read_csv("{}/y.dat".format(wds_data_folder), header=None, nrows=rows_cells * cols_cells, delim_whitespace=True, dtype=np.float32)
y_original = y_original.values.flatten()

# create a SVR object and specify the kernal and other parameters
svr_model = SVR(kernel='rbf', C=2000.0, gamma=0.3, epsilon=.1)
# build the SVR power distribution model
svr_model.fit(x_original, y_original)

# save the SVR model to a file
pickle.dump(svr_model, open("{}/{}".format(wds_data_folder,svr_model_filename), 'wb'))

# This is how to load SVR model from a file
# svr_model = pickle.load(open("{}/{}".format(wds_data_folder,svr_model_filename), 'rb'))
```
#### Run SUGGA method
```python
#SUGGA: support vector regression guided genetic algorithm
for i in range(0, run_times):  # run times
    print("run times {} ...".format(i))
    wfg.load_init_pop_NA("{}/init_{}.dat".format(init_pops_data_folder,i),"{}/init_{}_NA.dat".format(init_pops_data_folder,i))
    run_time, eta = wfg.sugga_genetic_alg(i,svr_model=svr_model,result_folder=sg_result_folder)
    result_arr[i, 0] = run_time
    result_arr[i, 1] = eta
time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = "{}/result_sugga_{}.dat".format(sg_result_folder,time_stamp)
np.savetxt(filename, result_arr, fmt='%f', delimiter="  ")
```
