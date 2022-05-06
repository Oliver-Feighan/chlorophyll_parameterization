%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1301_chromophore_17 TDDFT with cam-b3lyp functional

0 1
Mg   29.302   59.775   40.842
C   26.240   58.365   40.062
C   30.817   56.900   39.781
C   32.299   61.332   41.119
C   27.629   62.772   41.503
N   28.597   57.871   40.125
C   27.351   57.635   39.700
C   27.413   56.287   38.889
C   28.695   55.564   39.355
C   29.415   56.831   39.739
C   28.547   54.531   40.464
C   27.265   56.481   37.364
C   26.706   55.286   36.578
C   25.921   55.701   35.391
O   24.695   55.876   35.491
O   26.756   55.725   34.289
N   31.285   59.173   40.472
C   31.732   57.980   40.052
C   33.168   57.954   39.930
C   33.628   59.192   40.306
C   32.366   59.919   40.698
C   33.962   56.738   39.480
C   35.057   59.583   40.496
O   35.936   58.751   40.310
C   35.436   60.980   40.857
N   29.859   61.798   41.294
C   31.197   62.137   41.390
C   31.334   63.666   41.716
C   29.793   63.966   42.023
C   29.010   62.789   41.543
C   32.280   64.052   42.844
C   29.171   65.315   41.490
C   28.284   66.152   42.421
N   27.253   60.416   41.100
C   26.792   61.677   41.267
C   25.404   61.614   41.185
C   25.115   60.294   40.655
C   26.278   59.629   40.625
C   24.403   62.618   41.440
C   24.019   59.372   40.234
O   22.812   59.435   40.267
C   24.718   58.056   39.843
C   24.279   56.913   40.849
O   24.327   57.146   42.048
O   23.782   55.819   40.215
C   23.493   54.730   41.171
C   26.086   56.243   33.068
C   27.006   55.995   31.916
C   27.218   56.730   30.792
C   26.365   57.988   30.478
C   28.042   56.121   29.652
C   29.289   56.930   29.320
C   30.247   56.198   28.322
C   31.764   56.593   28.498
C   31.883   58.127   28.256
C   32.439   56.245   29.880
C   33.289   55.019   29.753
C   34.810   55.404   29.513
C   35.698   54.450   28.705
C   36.899   53.965   29.514
C   36.139   55.124   27.400
C   35.143   54.806   26.291
C   35.712   54.162   24.984
C   36.226   55.210   23.946
C   35.525   55.020   22.597
C   37.721   55.170   23.864
H   31.184   55.923   39.461
H   33.256   61.843   41.243
H   27.131   63.701   41.787
H   26.581   55.661   39.215
H   29.162   55.092   38.491
H   28.807   54.941   41.440
H   29.191   53.661   40.333
H   27.512   54.194   40.404
H   28.232   56.762   36.947
H   26.529   57.249   37.128
H   26.102   54.586   37.156
H   27.570   54.697   36.270
H   33.340   55.979   39.004
H   34.612   56.255   40.209
H   34.636   57.170   38.740
H   34.950   61.626   40.126
H   36.494   61.233   40.929
H   35.148   61.080   41.903
H   31.763   64.210   40.875
H   29.772   63.927   43.112
H   32.970   63.267   43.155
H   31.729   64.151   43.779
H   32.941   64.868   42.551
H   28.588   65.038   40.612
H   30.027   65.892   41.140
H   28.577   67.201   42.374
H   28.418   65.852   43.460
H   27.219   66.060   42.208
H   23.462   62.557   40.893
H   24.715   63.540   40.951
H   24.381   62.822   42.510
H   24.522   57.686   38.837
H   24.468   54.295   41.390
H   22.909   53.890   40.794
H   23.009   54.991   42.112
H   25.904   57.315   33.144
H   25.162   55.676   32.951
H   27.546   55.051   31.989
H   25.376   57.817   30.904
H   26.077   58.100   29.432
H   26.728   58.937   30.871
H   27.331   56.157   28.826
H   28.294   55.107   29.962
H   29.772   57.317   30.217
H   28.846   57.812   28.857
H   30.055   56.424   27.273
H   30.197   55.112   28.391
H   32.244   56.050   27.684
H   31.197   58.721   28.860
H   31.728   58.415   27.217
H   32.880   58.497   28.499
H   31.592   56.069   30.542
H   32.922   57.152   30.243
H   33.049   54.253   29.016
H   33.076   54.569   30.723
H   35.215   55.630   30.499
H   34.773   56.387   29.043
H   35.167   53.523   28.493
H   37.830   54.496   29.313
H   37.018   52.899   29.316
H   36.750   54.163   30.575
H   37.119   54.756   27.099
H   36.299   56.186   27.591
H   34.783   55.787   25.980
H   34.333   54.158   26.625
H   34.840   53.643   24.585
H   36.493   53.409   25.092
H   35.914   56.226   24.184
H   35.910   55.801   21.941
H   34.465   55.253   22.701
H   35.674   54.014   22.206
H   38.289   56.083   23.686
H   38.054   54.447   23.118
H   38.126   54.619   24.713

