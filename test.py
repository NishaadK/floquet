import jax.numpy as jnp
from scipy.optimize import linear_sum_assignment
import dynamiqs as dq
# overlaps = jnp.array([[0.9, 0.1, 0, 0], [0.1, 0.9, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.6, 0.4]])
# row_ind, col_ind = linear_sum_assignment(-overlaps)
# print(row_ind, col_ind)

# print(overlaps[:, col_ind])


# idx_exp_map = jnp.stack(jnp.meshgrid(
#                                     jnp.arange(3), 
#                                     jnp.arange(1, 5), 
#                                     indexing='ij'), 
#                                 axis=-1).reshape(-1, 2)

# print(idx_exp_map)
# a = dq.basis(3, 0) * (1 + 1j) / jnp.sqrt(2)
# b = dq.basis(3, 2) * (1 - 1j) / jnp.sqrt(2)

# qarr = dq.stack([a, b])
# print(type(qarr.to_jax()))

# a = jnp.array([jnp.array([1, 4]), jnp.array([2, 1])])
# b = jnp.array([jnp.array([2, 3]), jnp.array([1, 0])])

# # print(jnp.einsum("ij,jk->ik", a, b))

# print(jnp.argmax(b, axis=1))
# print(jnp.argmax(jnp.stack([a, b]), axis=2))

# # print(jnp.tile(a, (3, 1, 1)))

# # print(jnp.eye(5))

# c = [2, 3, 4, 5]
# print(c[2:4])

d = (10, 5)
f = jnp.stack(jnp.meshgrid(jnp.arange(d[0]), jnp.arange(d[1])), axis=2)
f = jnp.transpose(f, (1, 0, 2))
print(f)