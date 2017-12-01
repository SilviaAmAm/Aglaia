import tensorflow as tf


def dist_mat(X, n_atoms):
    """
    This function takes in a tensor containing all the cartesian coordinates of the atoms in a trajectory. Each line is
    a different configuration. It returns the upper triangular part of the distance squared matrix. (Note: couldnt use
    the distance instead of the distance squared, because the gradient of sqrt(X^2) approaches inf when X^2 goes to zero.
    This gives numerical instability when calculating gradients).

    :X: tensor with shape (n_samples, n_features)
    :n_samples: number of samples
    :n_atoms: number of atoms = int(n_features/3)
    :return: tensor of shape (n_samples, int(n_atoms * (n_atoms-1) * 0.5))
    """


    # This part generates the inverse matrix
    xyz_3d = tf.reshape(X, shape=(tf.shape(X)[0], n_atoms, 3))
    expanded_a = tf.expand_dims(xyz_3d, 2)
    expanded_b = tf.expand_dims(xyz_3d, 1)
    diff2 = tf.squared_difference(expanded_a, expanded_b, name='square_diff')
    diff2_sum = tf.reduce_sum(diff2, axis=3)
    # diff_sum = tf.sqrt(diff2_sum, name='diff')   # This has shape [n_samples, n_atoms, n_atoms]

    # This part takes the upper triangular part (no diagonal) of the distance matrix and flattens it to a vector
    ones = tf.ones_like(diff2_sum)
    mask_a = tf.matrix_band_part(ones, 0, -1)
    mask_b = tf.matrix_band_part(ones, 0, 0)
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool, name='mask') # Transfoorm into bool

    upper_triangular_conc = tf.boolean_mask(diff2_sum, mask, name='descript_concat')
    upper_triangular = tf.reshape(upper_triangular_conc, shape=(tf.shape(X)[0], int(n_atoms * (n_atoms-1) * 0.5)), name='descript')

    return upper_triangular

def inv_dist(X, n_atoms):
    """
    This function calculates the inverse distance squared matrix.

    :X: tensor with shape (n_samples, n_features)
    :n_samples: number of samples
    :n_atoms: number of atoms = int(n_features/3)
    :return: tensor of shape (n_samples, int(n_atoms * (n_atoms-1) * 0.5))
    """

    dist_matrix = dist_mat(X, n_atoms=n_atoms)

    inv_dist_matrix = 1/dist_matrix

    return inv_dist_matrix







