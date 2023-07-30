import torch
import numpy as np
from typing import Optional, Tuple
from functools import lru_cache


_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}


def _to_mat(pairs):
    mat = np.zeros((4, 4))
    for pair in pairs:
        key, value = pair
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat


_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])

_QUAT_MULTIPLY = np.zeros((4, 4, 4))
_QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]

_QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]

_QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]

_QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]

_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]

_CACHED_QUATS = {
    "_QTR_MAT": _QTR_MAT,
    "_QUAT_MULTIPLY": _QUAT_MULTIPLY,
    "_QUAT_MULTIPLY_BY_VEC": _QUAT_MULTIPLY_BY_VEC,
}


@lru_cache(maxsize=None)
def _get_quat(quat_key, dtype, device):
    return torch.tensor(_CACHED_QUATS[quat_key], dtype=dtype, device=device)


def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quat: [*, 4] quaternions
    Returns:
        [*, 3, 3] rotation matrices
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3]
    mat = _get_quat("_QTR_MAT", dtype=quat.dtype, device=quat.device)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))


def rot_to_quat(
    rot: torch.Tensor,
):
    if rot.shape[-2:] != (3, 3):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    k = [
        [
            xx + yy + zz,
            zy - yz,
            xz - zx,
            yx - xy,
        ],
        [
            zy - yz,
            xx - yy - zz,
            xy + yx,
            xz + zx,
        ],
        [
            xz - zx,
            xy + yx,
            yy - xx - zz,
            yz + zy,
        ],
        [
            yx - xy,
            xz + zx,
            yz + zy,
            zz - xx - yy,
        ],
    ]

    k = (1.0 / 3.0) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)

    _, vectors = torch.linalg.eigh(k)
    return vectors[..., -1]


@lru_cache(maxsize=None)
def identity_trans(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    trans = torch.zeros(
        (*batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad
    )
    return trans


@lru_cache(maxsize=None)
def identity_rot_mats(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)
    rots = rots.contiguous()

    return rots


def invert_rot_mat(rot_mat: torch.Tensor):
    return rot_mat.transpose(-1, -2)


def invert_quat(quat: torch.Tensor):
    quat_prime = quat.clone()
    quat_prime[..., 1:] *= -1
    inv = quat_prime / torch.sum(quat**2, dim=-1, keepdim=True)
    return inv


@lru_cache(maxsize=None)
def identity_quats(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    quat = torch.zeros(
        (*batch_dims, 4), dtype=dtype, device=device, requires_grad=requires_grad
    )

    with torch.no_grad():
        quat[..., 0] = 1

    return quat


class Rotation:
    """
    A 3D rotation. Depending on how the object is initialized, the
    rotation is represented by either a rotation matrix or a
    quaternion, though both formats are made available by helper functions.
    To simplify gradient computation, the underlying format of the
    rotation cannot be changed in-place. Like Rigid, the class is designed
    to mimic the behavior of a torch Tensor, almost as if each Rotation
    object were a tensor of rotations, in one format or another.
    """

    def __init__(
        self,
        rot_mats: Optional[torch.Tensor] = None,
        quats: Optional[torch.Tensor] = None,
        normalize_quats: bool = True,
    ):
        """
        Args:
            rot_mats:
                A [*, 3, 3] rotation matrix tensor. Mutually exclusive with
                quats
            quats:
                A [*, 4] quaternion. Mutually exclusive with rot_mats. If
                normalize_quats is not True, must be a unit quaternion
            normalize_quats:
                If quats is specified, whether to normalize quats
        """
        if (rot_mats is None and quats is None) or (
            rot_mats is not None and quats is not None
        ):
            raise ValueError("Exactly one input argument must be specified")

        if (rot_mats is not None and rot_mats.shape[-2:] != (3, 3)) or (
            quats is not None and quats.shape[-1] != 4
        ):
            raise ValueError("Incorrectly shaped rotation matrix or quaternion")

        # Force full-precision
        if quats is not None:
            quats = quats.to(dtype=torch.float32)
        if rot_mats is not None:
            rot_mats = rot_mats.to(dtype=torch.float32)

        if quats is not None and normalize_quats:
            quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)

        self._rot_mats = rot_mats
        self._quats = quats

    @staticmethod
    def identity(
        shape,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ) -> Rotation:
        """
        Returns an identity Rotation.

        Args:
            shape:
                The "shape" of the resulting Rotation object. See documentation
                for the shape property
            dtype:
                The torch dtype for the rotation
            device:
                The torch device for the new rotation
            requires_grad:
                Whether the underlying tensors in the new rotation object
                should require gradient computation
            fmt:
                One of "quat" or "rot_mat". Determines the underlying format
                of the new object's rotation
        Returns:
            A new identity rotation
        """
        if fmt == "rot_mat":
            rot_mats = identity_rot_mats(
                shape,
                dtype,
                device,
                requires_grad,
            )
            return Rotation(rot_mats=rot_mats, quats=None)
        elif fmt == "quat":
            quats = identity_quats(shape, dtype, device, requires_grad)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError(f"Invalid format: f{fmt}")
    
    @staticmethod
    def from_eula_angles(a, b, c):
        pass


class Rigid:
    """
    A class representing a rigid transformation. Little more than a wrapper
    around two objects: a Rotation object and a [*, 3] translation
    Designed to behave approximately like a single torch tensor with the
    shape of the shared batch dimensions of its component parts.
    """

    def __init__(
        self,
        rots: Optional[Rotation],
        trans: Optional[torch.Tensor],
    ):
        """
        Args:
            rots: A [*, 3, 3] rotation tensor
            trans: A corresponding [*, 3] translation tensor
        """
        # (we need device, dtype, etc. from at least one input)

        batch_dims, dtype, device, requires_grad = None, None, None, None
        if trans is not None:
            batch_dims = trans.shape[:-1]
            dtype = trans.dtype
            device = trans.device
            requires_grad = trans.requires_grad
        elif rots is not None:
            batch_dims = rots.shape
            dtype = rots.dtype
            device = rots.device
            requires_grad = rots.requires_grad
        else:
            raise ValueError("At least one input argument must be specified")

        if rots is None:
            rots = Rotation.identity(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )
        elif trans is None:
            trans = identity_trans(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )

        if (rots.shape != trans.shape[:-1]) or (rots.device != trans.device):
            raise ValueError("Rots and trans incompatible")

        # Force full precision. Happens to the rotations automatically.
        trans = trans.to(dtype=torch.float32)

        self._rots = rots
        self._trans = trans

    @staticmethod
    def identity(
        shape: Tuple[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ) -> Rigid:
        """
        Constructs an identity transformation.

        Args:
            shape:
                The desired shape
            dtype:
                The dtype of both internal tensors
            device:
                The device of both internal tensors
            requires_grad:
                Whether grad should be enabled for the internal tensors
        Returns:
            The identity transformation
        """
        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad, fmt=fmt),
            identity_trans(shape, dtype, device, requires_grad),
        )
