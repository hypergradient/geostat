# import geostat
# import geostat.mean as mn
# import geostat.kernel as krn
# from geostat import Parameters, GP
# p = Parameters(nugget=1., sill=1., beta=[4., 3., 2., 1.])
# kernel = krn.Noise(p.nugget)
# gp = GP(kernel=kernel)

# print(gp.mean)

# @geostat.featurizer()
# def trend_featurizer(x, y): return 1., x, y, x*y
# mean_function = mn.Trend(trend_featurizer, beta=p.beta)
# gp = GP(mean=mean_function, kernel=kernel)

# gp1 = GP(kernel=krn.Noise(p.nugget))
# gp2 = GP(mean=mean_function, kernel=krn.Delta(p.sill))
# combined_gp = gp1 + gp2
# print("Combined Mean: ", combined_gp.mean)  # <Trend object>
# print("Combined Kernel: ", combined_gp.kernel)  # <Stack object>

#-----------------------------------------------------------------#

# import tensorflow as tf
# from geostat.model import NormalizingFeaturizer

# # Define a simple featurization function
# def custom_featurizer(x, y):
#     return x, y, x * y

# # Sample location data
# locs = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# # Create the NormalizingFeaturizer
# norm_featurizer = NormalizingFeaturizer(custom_featurizer, locs)

# new_locs = tf.constant([[7.0, 8.0], [9.0, 10.0]])
# F_matrix = norm_featurizer(new_locs)
# print(F_matrix) # F_matrix will contain normalized features with an additional intercept column
# # tf.Tensor(
# # [[1.        2.4494898 2.4494898 3.5676992]
# #  [1.        3.6742349 3.6742349 6.50242  ]], shape=(2, 4), dtype=float32)

#-----------------------------------------------------------------#

# import tensorflow as tf
# from geostat.model import Featurizer

# # Define a custom featurization function
# def simple_featurizer(x, y):
#     return x, y, x * y

# # Initialize the Featurizer
# featurizer = Featurizer(simple_featurizer)

# locs = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# F_matrix = featurizer(locs)
# print(F_matrix) # F_matrix will contain the features: (x, y, x*y) for each location
# # tf.Tensor(
# # [[ 1.  2.  2.]
# #  [ 3.  4. 12.]
# #  [ 5.  6. 30.]], shape=(3, 3), dtype=float32)


# featurizer_no_feat = Featurizer(None)
# F_matrix = featurizer_no_feat(locs)
# print(F_matrix) # Since no featurization function is provided, F_matrix will have shape (3, 0)
# # tf.Tensor([], shape=(3, 0), dtype=float32)

#-----------------------------------------------------------------#

# from geostat import GP, Model, Parameters
# from geostat.kernel import Noise
# import numpy as np

# # Create parameters.
# p = Parameters(nugget=1.)

# # Define the Gaussian Process and the model
# gp = GP(kernel=Noise(nugget=p.nugget))
# locs = np.array([[0.0, 1.0], [1.0, 2.0]])
# vals = np.array([1.0, 2.0])
# model = Model(gp=gp, locs=locs, vals=vals)

#-----------------------------------------------------------------#

# from geostat import GP, Model, Parameters
# from geostat.kernel import Noise

# # Create parameters.
# p = Parameters(nugget=1.)

# # Create model
# kernel = Noise(nugget=p.nugget)
# model = Model(GP(0, kernel))

# # Update parameters
# model.set(nugget=0.5)

#-----------------------------------------------------------------#

# from geostat import GP, Model, Parameters
# from geostat.kernel import Noise
# import numpy as np

# # Create parameters.
# p = Parameters(nugget=1.)

# # Create model
# kernel = Noise(nugget=p.nugget)
# model = Model(GP(0, kernel))

# # Fit model
# locs = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
# vals = np.array([10.0, 15.0, 20.0])
# model.fit(locs, vals, step_size=0.05, iters=500)
# # [iter    50 ll -63.71 time  2.72 reg  0.00 nugget  6.37]
# # [iter   100 ll -32.94 time  0.25 reg  0.00 nugget 13.97]
# # [iter   150 ll -23.56 time  0.25 reg  0.00 nugget 22.65]
# # [iter   200 ll -19.26 time  0.25 reg  0.00 nugget 32.27]
# # [iter   250 ll -16.92 time  0.25 reg  0.00 nugget 42.63]
# # [iter   300 ll -15.52 time  0.24 reg  0.00 nugget 53.50]
# # [iter   350 ll -14.63 time  0.24 reg  0.00 nugget 64.71]
# # [iter   400 ll -14.03 time  0.24 reg  0.00 nugget 76.10]
# # [iter   450 ll -13.61 time  0.25 reg  0.00 nugget 87.52]
# # [iter   500 ll -13.32 time  0.24 reg  0.00 nugget 98.85]

# cats = np.array([1, 1, 2])
# model.fit(locs, vals, cats=cats, step_size=0.01, iters=300)
# # [iter    30 ll -12.84 time  0.25 reg  0.00 nugget 131.53]
# # [iter    60 ll -12.62 time  0.15 reg  0.00 nugget 164.41]
# # [iter    90 ll -12.53 time  0.16 reg  0.00 nugget 191.70]
# # [iter   120 ll -12.50 time  0.16 reg  0.00 nugget 211.74]
# # [iter   150 ll -12.49 time  0.15 reg  0.00 nugget 225.07]
# # [iter   180 ll -12.49 time  0.16 reg  0.00 nugget 233.15]
# # [iter   210 ll -12.49 time  0.15 reg  0.00 nugget 237.64]
# # [iter   240 ll -12.49 time  0.15 reg  0.00 nugget 239.92]
# # [iter   270 ll -12.49 time  0.15 reg  0.00 nugget 240.98]
# # [iter   300 ll -12.49 time  0.15 reg  0.00 nugget 241.42]

#-----------------------------------------------------------------#

# from geostat import GP, Model, Parameters
# from geostat.kernel import Noise
# import numpy as np

# # Create parameters.
# p = Parameters(nugget=1.)

# # Create model
# kernel = Noise(nugget=p.nugget)
# model = Model(GP(0, kernel))

# # Generate values based on locs
# locs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# model.generate(locs)
# generated_vals = model.vals  # Access the generated values
# print(generated_vals)
# # [0.45151083 1.23276189 0.3822659 ] (Values are non-deterministic)

#-----------------------------------------------------------------#

# from geostat import GP, Model, Parameters
# from geostat.kernel import SquaredExponential
# import numpy as np

# # Create parameters.
# p = Parameters(sill=1.0, range=2.0)

# # Create model
# kernel = SquaredExponential(sill=p.sill, range=p.range)
# model = Model(GP(0, kernel))

# # Fit model
# locs = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
# vals = np.array([10.0, 15.0, 20.0])
# model.fit(locs, vals, step_size=0.05, iters=500)
# # [iter    50 ll -40.27 time  2.29 reg  0.00 sill  6.35 range  1.96]
# # [iter   100 ll -21.79 time  0.40 reg  0.00 sill 13.84 range  2.18]
# # [iter   150 ll -16.17 time  0.39 reg  0.00 sill 22.31 range  2.44]
# # [iter   200 ll -13.55 time  0.39 reg  0.00 sill 31.75 range  2.76]
# # [iter   250 ll -12.08 time  0.38 reg  0.00 sill 42.08 range  3.12]
# # [iter   300 ll -11.14 time  0.38 reg  0.00 sill 53.29 range  3.48]
# # [iter   350 ll -10.50 time  0.38 reg  0.00 sill 65.36 range  3.85]
# # [iter   400 ll -10.05 time  0.39 reg  0.00 sill 78.29 range  4.22]
# # [iter   450 ll -9.70 time  0.39 reg  0.00 sill 92.07 range  4.59]
# # [iter   500 ll -9.43 time  0.39 reg  0.00 sill 106.70 range  4.95]

# # Run predictions
# locs2 = np.array([[1.5, 1.5], [2.5, 4.0]])
# mean, variance = model.predict(locs2)
# print(mean)
# # [ 9.89839798 18.77077269]
# print(variance)
# # [2.1572128  0.54444738]

#-----------------------------------------------------------------#
