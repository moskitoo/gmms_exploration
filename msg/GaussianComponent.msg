# A single Gaussian component in the mixture model
# Mean of the Gaussian (3D: x, y, z)
float64[3] mean

# Covariance matrix stored as a flat array (4x4 = 16 elements)
# Row-major order: [cov(0,0), cov(0,1), ..., cov(3,3)]
float64[9] covariance

# Weight/prior probability of this component
float64 weight

float64 uncertainty

int32 fusion_count

int32 observation_count

float64 last_displacement