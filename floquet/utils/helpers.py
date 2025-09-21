
import jax.numpy as jnp
from jax import jit, vmap, Array
from scipy.optimize import linear_sum_assignment
from jaxtyping import Complex, Float
from typing import List
from dynamiqs import QArray
import dynamiqs as dq

# @jit
def overlap(state_set_0: QArray, state_set_1: QArray) -> Array:

    state_set_0 = jnp.squeeze(dq.to_jax(state_set_0))
    state_set_1 = jnp.squeeze(dq.to_jax(state_set_1))

    return jnp.abs(jnp.einsum("ij,kj->ik", state_set_0.conj(), state_set_1))



@jit
def assign_states(overlap_matrix: Float[Array, "dim1 dim2"]) -> Array:
    # _, max_overlap_indices = linear_sum_assignment(-overlap_matrix)  # MWPM solve

    max_overlap_indices = jnp.argmax(overlap_matrix, axis=1)
    return max_overlap_indices
    # return jnp.array(max_overlap_indices)

