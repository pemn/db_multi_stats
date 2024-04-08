#!python
# Copyright 2019 Vale
# voxelbase
# v1.0 05/2019 paulo.ernesto

import numpy as np
from skimage.transform import matrix_transform, AffineTransform

def sanitize_cell_size(cell_size):
  bearing = None
  if isinstance(cell_size, str):
    import re
    size = re.split('[,;_]', cell_size)
    if len(size) > 3:
      bearing = float(size[3])
      size = size[:3]
    while len(size) < 3:
      size.append(size[-1])
    size = np.asfarray(size)
  else:
    size = np.full(3, cell_size or 10, dtype=np.float_)
  return size, bearing

class VoxelBase(np.ma.MaskedArray):
  def __new__(cls, size, shape = None, origin = None, bearing = None):
    if not hasattr(size, '__len__'):
      size = np.full(3, size)
    vdim = np.arange(len(size))
    if shape is None:
      shape = np.ones(len(size))
    elif not hasattr(shape, '__len__'):
      shape = np.full(3, shape)
    else:
      shape = np.array(shape, dtype=np.int_)

    if origin is None:
      origin = np.zeros(len(size))

    rotation = None
    if bearing is not None:
      rotation = np.deg2rad(- (bearing - 90))
    print("rotation",np.rad2deg(rotation))

    self = super().__new__(cls, np.ndarray(shape), np.ones(np.prod(shape), np.bool_))
    self._dim = np.array(size)
    self._ndim = len(vdim)
    self._bb0 = origin
    self._aft = AffineTransform(scale=size[:2], rotation=rotation, translation=origin[:2])
    self._aft_inv = np.linalg.inv(self._aft.params)
    print("cell size",self._dim)
    print("origin",self._bb0)
    print("shape",self.shape)
    return(self)
  def __repr__(self):
    return repr(self.to_df().describe())

  # return the ijk of a voxel using our fixed voxel size instead of the voxel own size
  def ijk(self, xyz, mode = 'c'):
    ijk = []
    if np.ndim(xyz) == 1:
      ij = matrix_transform(xyz[:2], self._aft_inv)
      ijk.extend(ij[0])
      ijk.append(xyz[2] / self._dim[2] - self._bb0[2])
    else:
      ij = matrix_transform(xyz[:,:2], self._aft_inv)
      k = ijk[:, 2] / self._dim[2] - self._bb0[2]
      ijk = np.concatenate((ij, k.reshape((len(k),1))), 1)
    if mode == 'c':
      ijk = np.subtract(ijk, 0.5)
    if mode == 'bb1':
      ijk = np.subtract(ijk, 1)
    # else: bb0
    return ijk

  def ijk_old(self, xyz):
    #(block_centre, block_length) = Voxel.block(self._data.ix[c])
    #return tuple(int((xyz[i] - self._o0[i]) // self._bl[i]) for i in range(3))
    # return np.divide(np.subtract(xyz, self._o0), self._bl).astype(np.int_)
    return tuple(np.array(np.subtract(np.resize(xyz, self._ndim), self._bb0) // self._dim, dtype=np.int_))

  def xyz_bmf(self, ijk, bm):
    return bm.to_world(*self.xyz(ijk))

  def xyz_old(self, ijk, mode = 'c'):
    '''
    c = centroid
    box0 = lower left corner
    box1 = upper right corner
    '''

    c = np.multiply(np.resize(ijk, self._ndim), self._dim) + self._bb0
    if mode == 'c':
      return c + np.multiply(self._dim, 0.5)
    if mode == 'box0':
      return c
    if mode == 'box1':
      return c + self._dim

  def xyz(self, ijk, mode = 'c'):
    xyz = []
    if mode == 'c':
      ijk = np.add(ijk, 0.5)
    if mode == 'bb1':
      ijk = np.add(ijk, 1)
    # else: bb0
    if np.ndim(ijk) == 1:
      xy = matrix_transform(ijk[:2], self._aft.params)
      xyz.extend(xy[0])
      xyz.append(ijk[2] * self._dim[2] + self._bb0[2])
    else:
      xy = matrix_transform(ijk[:,:2], self._aft.params)
      #xyz.extend(xy)
      z = ijk[:, 2] * self._dim[2] + self._bb0[2]
      xyz = np.concatenate((xy, z.reshape((len(z),1))), 1)

    return xyz

  def bb(self, ijk):
    '''
    local bounding box
    '''
    box0 = np.multiply(np.resize(ijk, self._ndim), self._dim)
    box1 = np.add(box0, self._dim)
    return(box0, box1)

  def show(self):
    vbn = np.add(self.shape, 1)
    vbi = np.indices(vbn)
    vbc = self.xyz(vbi.transpose(1,2,3,0).reshape((np.prod(vbn), 3)))

    vbv = np.moveaxis(np.reshape(vbc, np.concatenate((vbn, [3]))), 3, 0)

    # vb.fill(True)
    # vb.mask.fill(False)
    self[:] = True

    fc = np.linspace(0, 1, np.prod(self.shape)).reshape(self.shape)
    fc = np.stack((fc, fc, fc), 3)

    plt.subplot(121, projection='3d')
    plt.gca().voxels(self, facecolors=fc)
    plt.subplot(122, projection='3d')

    plt.gca().voxels(*vbv, self, facecolors=fc)
    plt.show()

  @classmethod
  def from_bmf(cls, bm):
    n_schema = bm.model_n_schemas()-1
    size = np.resize(bm.model_schema_size(n_schema), 3)
    shape = bm.model_schema_dimensions(n_schema)
    s_origin = bm.model_schema_extent(n_schema)
    b_origin = bm.model_origin()
    t_origin = np.add(b_origin, s_origin[:3])
    orientation = bm.model_orientation()
    if shape[2] == 1:
      shape[2] = size[2]
      size[2] = 1

    return(cls(size, shape, t_origin, orientation[0]))

  @classmethod
  def from_schema(cls, df, cell_size = None):
    size, bearing = sanitize_cell_size(cell_size)

    from _gui import pd_detect_xyz, pd_detect_rr, getRectangleSchema
    xyz = pd_detect_xyz(df)
    rr = pd_detect_rr(df, xyz)
    #print(list(rr))
    if bearing is None:
      origin2d, dims2d, r_bearing = getRectangleSchema(rr, size)
      bearing = r_bearing
      origin = np.append(origin2d, df[xyz[2]].min())
      shapez = max(1, np.ceil(np.abs(np.subtract(df[xyz[2]].max(), df[xyz[2]].min()) / size[2])))
      shape = np.append(dims2d, shapez)
    else:
      #origin2d = np.min(rr, 0)
      #dims2d = np.ceil(np.divide(np.subtract(np.max(rr, 0), np.min(rr, 0)), size[:2]))
      origin = np.min(df[xyz], 0)
      shape = np.max([np.asarray(np.ceil(np.divide(np.subtract(np.max(df[xyz], 0), np.min(df[xyz], 0)), size)), np.int_), np.ones(3, dtype=np.int_)], 0)

    return(cls(size, shape, origin, bearing))

  def to_flat_ijk(self):
    ' flat table (list of lists) of all voxel ijk indices'
    return np.indices(self.shape).transpose(1,2,3,0).reshape((np.prod(self.shape), 3))

  def to_flat_xyz(self):
    ' flat table (list of lists) of all voxel xyz coordinates '
    return self.xyz(self.to_flat_ijk())

  def to_bm(self, output):
    import vulcan
    bm = vulcan.block_model()
    xyz0 = np.zeros(3)
    xyz1 = self.shape * self._dim
    xyzn = self.shape
    xyzo = self._bb0
    bm.create_regular(output, *xyz0, *xyz1, int(xyzn[0]), int(xyzn[1]), int(xyzn[2]))
    bm.set_model_origin(*xyzo)
    bm.write()
    print("index model")
    bm.index_model()
    bm.write()
    return bm

  def to_df(self):
    import pandas as pd
    return pd.DataFrame(self.to_flat_xyz(), columns=['x','y','z'])


if __name__=="__main__":
  import sys
  if len(sys.argv) > 1:
    from _gui import pd_load_dataframe
    df = pd_load_dataframe(sys.argv[1])
    grid = VoxelBase.from_schema(df, 50)
    import pandas as pd
    print(pd.DataFrame(grid.to_flat_xyz()))
