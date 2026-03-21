"""
    Mathematical auxiliary functions.
"""

__all__ = ['calc_pad_value', 'encode_vectors_via_pca', 'decode_vectors_via_pca']

import numpy as np


def calc_pad_value(src_value: int,
                   dst_value: int) -> tuple[int, int]:
    """
    Calculate left and right padding values for the source and destination sizes.

    Parameters
    ----------
    src_value : int
        Source number.
    dst_value : int
        Destination number.

    Returns
    -------
    int
        Left padding.
    int
        Right padding.
    """
    if dst_value < src_value:
        raise Exception("Destination value is smaller than source one")
    if src_value == dst_value:
        pad_left = 0
        pad_right = 0
    else:
        pad_value = dst_value - src_value
        pad_left = pad_value // 2
        pad_right = pad_value - pad_left
    return pad_left, pad_right


def encode_vectors_via_pca(vectors: np.ndarray,
                           pca_params: dict[str, np.ndarray],
                           calc_whitening: bool = False,
                           return_both: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Encode (project to subspace) vectors via PCA.

    Parameters
    ----------
    vectors : np.ndarray
        Input floating-point vectors. This is an (n, m) matrix,
        where n is the number of vectors and m is the vector length.
    pca_params : dict[str, np.ndarray]
        PCA params.
    calc_whitening : bool, default False
        Whether to encode with whitening.
    return_both : bool, default False
        Whether to return both encoded representations (with and without whitening).

    Returns
    -------
    np.ndarray or tuple(np.ndarray, np.ndarray)
        Encoded vectors or two sets of encoded vectors (without whitening and with).
    """
    vp = np.divide(vectors - pca_params["mean1"], pca_params["std1"],
                   where=(pca_params["std1"] != 0.0)).dot(pca_params["pca_m"])
    if not calc_whitening:
        return vp
    else:
        vw = vp / pca_params["std2"]
        if not return_both:
            return vw
        else:
            return vp, vw


def decode_vectors_via_pca(vectors: np.ndarray,
                           pca_params: dict[str, np.ndarray],
                           used_whitening: bool = False) -> np.ndarray:
    """
    Decode (reconstruct from subspace projections) vectors via PCA.

    Parameters
    ----------
    vectors : np.ndarray
        Input floating-point vectors. This is an (n, m) matrix,
        where n is the number of vectors and m is the vector length.
    pca_params : dict[str, np.ndarray]
        PCA params.
    used_whitening : bool, default False
        Whether whitening was used during encoding.

    Returns
    -------
    np.ndarray
        Decoded vectors.
    """
    vp = vectors * pca_params["std2"] if used_whitening else vectors
    vp = vp.dot(pca_params["pca_m"].transpose()) * pca_params["std1"] + pca_params["mean1"]
    return vp
