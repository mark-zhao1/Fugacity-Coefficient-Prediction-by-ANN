# Ln_phi_model
A Proxy Peng-Robinson EOS For Rapid Multiphase Flash
ANN prediction of fugacity coefficient from {Am, Bm, Bi, sum_xjAij} attributes.

Author: Mark Zhao
Questions to: mark.zhao@utexas.edu

# Data_generation generates training instances.  
    Specify ranges of T, P, x space of training data.  
    Stores in .csv, hdf5, or pickle format.  
    
# Domain_exploration visualizes the training data.  
    View discontinuous or non-smooth data.  
    Create and manipulate Pandas DataFrames for graphical visualization.
    Simple data preprocessing.  
    
# Trains ANN models to predict fugacity coefficient from Am, Bm, Bi, sum_xjAij.  
    Tensorflow training script to be used in a Jupyter Notebook: instantiate 'train' class in train.py.  
    Feature engineering done in Jupyter Notebook.  
    Tensorboard visualization.
    Data generation step sometimes included in experiment directory.
    Some workflows redacted for SPE paper and conferences.

# Applied Model in Stability Analysis and Phase Split
    ANN ln_phi model implemented to replace equation-of-state calculations.
    Redacted in preparation for SPE Reservoir Simulation Conference.
    
# Weekly Reports  
    Redacted.
