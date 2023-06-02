import numpy as np

def get_data():
    """Hard-coded data from Table 4 "SYMBOLIC DATA ANALYSIS: DEFINITIONS AND EXAMPLES"
    """
    low = np.array([[90,50,44], 
                    [90,70,60],
                    [140,90,56],
                    [110,80,70],
                    [90,50,54],
                    [134,80,70],
                    [130,76,72],
                    [110,70,76],
                    [138,90,86],
                    [110,78,86]])
    
    hi  = np.array([[110,70,68], 
                    [130,90,72],
                    [180,100,90],
                    [142,108,112],
                    [100,70,72],
                    [142,110,100],
                    [160,90,100],
                    [190,110,98],
                    [180,110,96],
                    [150,100,100]])
    arr = np.dstack([low,hi])
    x = arr[:,:2,:]
    y = np.expand_dims(arr[:,2,:], axis=1)

    return x,y

def check_negative_interval(arr):
    """Check if interval matrix contains negative interval

    Args:
        array (np.ndarray): interval matrix with dimension (n x m x 2)

    Returns:
        bool: True or false that the interval matrix contains negative interval
    """
    diff = arr[:,:,1] - arr[:,:,0]
    return np.any(diff<0)


if __name__ == "__main__":
    x,y = get_data()
    print(check_negative_interval(x))
