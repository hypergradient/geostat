import optax
import jax
import jax.numpy as jnp
def f(x): return jnp.sum(x ** 2)  # simple quadratic function
solver = optax.adam(learning_rate=0.1)
params = jnp.array([1., 2., 3.])
print('Objective function: ', f(params))
opt_state = solver.init(params)
for _ in range(50):
 grad = jax.grad(f)(params)
 updates, opt_state = solver.update(grad, opt_state, params)
 params = optax.apply_updates(params, updates)
 print('params: ', params)
 print('Objective function: {:.2E}'.format(f(params)))