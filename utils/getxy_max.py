import numpy as np

def getxy_max(input_array: np.array, n: int, min_dist: int = 3):
    """[Helper function to get n highest points in 2D array keeping a minimum distance between hits]
    
    Arguments:
        input_array {np.array} -- [2D array, typically an image]
        n {int} -- [number of points]
    
    Keyword Arguments:
        min_dist {int} -- [minimun distance between hits] (default: {3})
    
    Returns:
        [list] -- [list of x, y tuples with indices for hotspots]
    """
    # check for invalid inputs and copy array
    assert len(input_array.shape) in [2, 3], 'invalid input dimensions'
    if len(input_array.shape) == 3:
        array = input_array.copy()[0, :, :]
    else:
        array = input_array.copy()
    
    # get shape
    n_rows, n_cols = array.shape

    # get sorted indices
    sorted_idcs = np.argsort(-array, axis=None)

    # get n locations with the highest value, keeping a minimum distance between hits
    out_xy = []
    idx = 0
    while len(out_xy) < n and idx < len(sorted_idcs):
        # add maximum
        row = sorted_idcs[idx] // n_cols
        col = sorted_idcs[idx] % n_cols
        
        # skip value if it is too close to a previous point
        if array[row, col] == 0:
            idx += 1
            continue
        
        # add point to output list and mask out nearby points otherwise
        else:
            idx += 1
            out_xy.append((row, col))
            array[max(0, row - min_dist): min(n_rows, row + min_dist),
                max(0, col - min_dist): min(n_cols, col + min_dist)] = 0
    
    # return indices
    return out_xy






