import jax.numpy as jnp
from scipy.optimize import linear_sum_assignment
import dynamiqs as dq
import numpy as np
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

# d = (10, 5)
# f = jnp.stack(jnp.meshgrid(jnp.arange(d[0]), jnp.arange(d[1])), axis=2)
# f = jnp.transpose(f, (1, 0, 2))
# print(f)

# a = jnp.array([1, 2])
# b = jnp.array([2, 4])

# print(jnp.einsum('ij, kj -> ik', a, b))

omega_d_idxs = jnp.arange(5)
amp_idxs = jnp.arange(10)
# omega_d_amp_idx_grid = jnp.stack(jnp.meshgrid(omega_d_idxs, amp_idxs), axis=-1)
# omega_d_amp_idx_grid = jnp.transpose(omega_d_amp_idx_grid, (1, 0, 2))
# print(omega_d_amp_idx_grid)

freq_amp_shape = (5, 10)
fit_range_fraction = 0.4
num_fit_ranges = int(jnp.ceil(1 / fit_range_fraction))
num_amp_pts_per_range = int(jnp.floor(freq_amp_shape[-1] / num_fit_ranges)
)

omega_idx_set = jnp.array([0, freq_amp_shape[-2]])
amp_idx_set = jnp.array([jnp.array([i*num_amp_pts_per_range, int(jnp.min(jnp.array([(i+1) * num_amp_pts_per_range, 10])))])
                for i in range(num_fit_ranges + 1)])
# amp_idx_set = [[i*num_amp_pts_per_range, jnp.min((i+1) * num_amp_pts_per_range, freq_amp_shape[-1])] 
#                 for i in range(num_fit_ranges)]

print(jnp.stack([jnp.repeat(omega_idx_set[None, :], amp_idx_set.shape[0], axis=0), amp_idx_set], axis=1))

# print(jnp.meshgrid(jnp.array(omega_idx_set), jnp.array(amp_idx_set)))