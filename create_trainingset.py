import pandas
import numpy
import os
import gdal


def get_patches(raster_dir, vector_df, lon, lat, patch_sizes, labels):
    """
    Generates multi-band patches at different scales around vector points to use as a training set.

    Input:
        raster_dir : directory with raster images (.tif) we wish to extract patches from.
        vector_df : pandas data.frame with training points with latitude, longitude, classification label and source
            raster layer.
        lon : column in vector_df containing longitude component.
        lat : column in vector_df containing latitude component.
        patch_sizes : list with pyramid dimensions in multi-band output, first element is used as base dimension,
            subsequent elements must be larger than the previous element and will be re-sized to match patch_sizes[0]
        labels : list with training set labels

    Output:
        folder with extracted multi-band training images separated in subfolders by intended label
    """
    #TODO: check for increasing size in patch_sizes
    #TODO: create output directory on current path and subdirectories based on training labels
    rasters = []
    for path, _, files in os.walk(raster_dir):
        for filename in files:
            filename_lower = filename.lower()
            if not (any(filename_lower.endswith('.tif'))):
                print('{} is not a valid scene.'.format(filename_lower))
                continue
            rasters.append(os.path.join(path, filename))
    for i, rs in enumerate(rasters):

        presValues = []
        gdata = gdal.Open('{}'.format(rs))
        gt = gdata.GetGeoTransform()
        band = gdata.GetRasterBand(1)
        nodata = band.GetNoDataValue()

        # gt(2) and gt(4) coefficients are zero, and the gt(1) is pixel width, and gt(5) is pixel height.
        # The (gt(0),gt(3)) position is the top left corner of the top left pixel of the raster.
        x0, y0, w, h = gt[0], gt[3], gt[1], gt[5]

        data = band.ReadAsArray().astype(numpy.float)
        #TODO: add padding to raster data based on the last element patch_sizes
        # free up memory
        del gdata
        # iterate through the points
        for p in vector_df.iterrows():
            x = int((p[1][lon] - x0)/w)
            y = int((p[1][lat] - y0)/h)
            for scale in patch_sizes:
                try:
                   if data[y, x] != nodata:
                      #TODO: save patch image to correct subfolder based on label
                      data[y, x]
                except:
                    pass

        del data, band
