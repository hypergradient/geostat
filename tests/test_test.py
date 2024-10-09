import jax.numpy as jnp
from geostat import Parameters, GP, Model, Trend



def test():

    p = Parameters(nugget=1.)
    
    print("After Initialization: ", p)

    print(p.nugget)

    p.nugget.create_jax_variable()

    #p.nugget.underlying = jnp.array([2.0])
    #p.nugget.update_value()

    print(p.nugget)
    p.nugget.update_bounds(0, 10)
    print(p.nugget)
    p.nugget.create_jax_variable()
    print(p.nugget)
    p.nugget.update_value()
    print(p.nugget)

    print(p.nugget.surface())
    p.nugget.underlying = 10.0
    print(p.nugget)
    p.nugget.update_value()
    print(p.nugget)
