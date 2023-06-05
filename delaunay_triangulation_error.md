# Documentation: Handling Errors in Delaunay Triangulation

This page describes a common error that can occur during Delaunay triangulation when generating contour plots, and provides potential solutions.

## Understanding the Error

The error message typically manifests as a `QhullError`, indicating a failure in the underlying Qhull library which is responsible for the computation of Delaunay triangulations and Voronoi diagrams.

```plaintext
scipy.spatial._qhull.QhullError: QH6154 Qhull precision error: Initial simplex is flat (facet 1 is coplanar with the interior point)
```

This error arises when the data provided for the triangulation is either co-planar or cannot be used to construct a 3D simplex, a prerequisite for creating a contour plot.

## Potential Solutions

### 1. Adding Small Random Noise to Your Data

A small random noise or jitter can be added to make the dataset non-coplanar, and thus suitable for Delaunay triangulation. However, this approach could potentially distort your data, so use it cautiously.

```python
jitter = 0.001  # small random noise
xs += np.random.uniform(-jitter, jitter, xs.shape)
ys += np.random.uniform(-jitter, jitter, ys.shape)
zs += np.random.uniform(-jitter, jitter, zs.shape)
```

### 2. Filtering out Near-Collinear Points

Sequences of points that are very close to being on a straight line (collinear) can cause issues with Delaunay triangulation. A preprocessing step can be written to remove these points from your data.

### 3. Switching to a Different Interpolation Method

The error arises from the 'CloughTocher2DInterpolator' method used by scipy's `griddata` function for data interpolation. Using a different method like 'linear' or 'nearest' which do not require Delaunay triangulation can be an alternative solution.

```python
grid_z = scipy_interp.griddata(
    points, values, (grid_x, grid_y), method='nearest')
```

### 4. Transforming Data into a Coordinate Plane

If your data is inherently lower-dimensional, you can transform your data into a coordinate plane. This is a more complex solution but may be necessary for some data sets. This could be done by performing a Principal Component Analysis (PCA), and then deleting the other coordinates.

Remember that while these solutions can potentially resolve the error, they might introduce some degree of distortion or bias in your data, so use them with caution.
