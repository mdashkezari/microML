# Marine Microbe Model 

## Spatial Distributions of Prochlorococcus, Synechococcus,  Picoeukaryotes, and Heterotrophic Bacteria Abundances Estimated by Machine Learning Algorithms

This project compiles 4 decades of direct measurements of marine phytoplankton abundances at global scale, ranging from ocean surface to ~5900 m deep, with over 428,000 discrete samples.

<img src="media/spatial_dist.png" alt="spatial_dist" width="500"/>


All measurements are then augmented with a large number of contemporaneous environmental parameters such as temperature, salinity, nutrients, etc. Once data is processed, the ML model uses the environmental parameters (or a transformed version of them) to predict the organism abundances. Below, are the model predictions during a 8-year period.




<video width="640" height="360" controls>
  <source src="media/prochlorococcus_abundance.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## Installation:
```
conda create -n mml python=3.11
conda activate mml
conda install -c conda-forge gdal
# cd to where pyproject.toml is
pip install -e .  

# copy untrack/assets/natural_earth to cartopy maps dirs
```

## Usage:
```
$ cd microML
$ pip install -e .
$ cd microML
$ python main.py 
```

