# ML_for_reionization

These are a series of machine learning notebooks using a suit of 21cmfaster runs, which calculate the redshift of reionization of each cell in a comoving volume, as its data. The outputs of the 21cmfaster runs are 512^3 resolution boxes representing a (400Mpc)^3 comoving volume of the universe. All were run with the same initial paramaters.

There are 3 data types used in these notebooks which were used as training, validation, and test data:
- the Lagrangian density field scaled to z=0.0
- the Eulerian density field at z=10.0
- the redshift of reionization

Data format:
The input and reionization data is binary float32 format. Simply read in 512^3 float32's and reshape to an array of shape=(512, 512, 512). 

The data is reshaped into 4096 cubes of array shape (32, 32, 32) to be used in each network.

Networks are labeled by their type and the number of convolutional blocks within them.
