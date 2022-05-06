%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1251_chromophore_14 TDDFT with cam-b3lyp functional

0 1
Mg   45.735   44.564   43.839
C   42.390   43.458   43.640
C   46.633   41.336   43.044
C   48.783   45.659   43.333
C   44.547   47.973   44.032
N   44.621   42.631   43.362
C   43.253   42.457   43.385
C   42.860   41.037   42.988
C   44.293   40.318   42.946
C   45.247   41.473   43.228
C   44.451   39.148   43.939
C   42.126   40.857   41.606
C   42.450   41.900   40.443
C   42.424   41.209   39.070
O   41.788   40.239   38.733
O   43.111   42.005   38.148
N   47.444   43.627   43.165
C   47.611   42.323   42.898
C   48.984   42.072   42.535
C   49.613   43.352   42.460
C   48.627   44.319   42.950
C   49.563   40.737   42.103
C   50.993   43.578   41.993
O   51.755   42.640   41.737
C   51.462   44.970   41.636
N   46.536   46.585   43.512
C   47.882   46.671   43.658
C   48.343   48.026   44.107
C   47.083   48.892   44.165
C   45.972   47.758   43.883
C   49.066   47.993   45.422
C   47.121   50.009   43.120
C   46.420   51.370   43.451
N   43.961   45.525   43.830
C   43.589   46.885   43.967
C   42.165   47.035   43.940
C   41.659   45.746   43.753
C   42.763   44.872   43.744
C   41.348   48.287   44.059
C   40.430   44.968   43.719
O   39.266   45.379   43.641
C   40.887   43.473   43.652
C   40.231   42.837   44.819
O   39.171   42.194   44.870
O   40.917   43.179   45.928
C   40.436   42.594   47.242
C   42.814   41.709   36.714
C   43.406   42.656   35.770
C   44.525   42.498   34.919
C   45.586   41.413   35.200
C   44.885   43.585   33.752
C   44.753   43.052   32.326
C   43.724   43.701   31.429
C   44.376   44.085   30.071
C   44.586   45.589   29.948
C   43.561   43.441   28.867
C   44.588   43.367   27.651
C   44.029   43.435   26.277
C   44.429   44.705   25.482
C   43.531   45.943   25.814
C   44.563   44.499   23.946
C   45.753   45.041   23.268
C   46.906   43.973   23.246
C   48.262   44.548   22.730
C   49.117   43.525   21.882
C   49.139   45.089   23.864
H   46.920   40.347   42.683
H   49.847   45.870   43.457
H   44.125   48.972   44.159
H   42.230   40.566   43.743
H   44.565   40.083   41.917
H   45.073   38.384   43.471
H   43.454   38.728   44.073
H   44.808   39.490   44.910
H   41.062   40.844   41.838
H   42.263   39.837   41.245
H   43.467   42.230   40.659
H   41.668   42.660   40.445
H   50.124   40.785   41.169
H   48.745   40.063   41.851
H   50.171   40.236   42.857
H   51.680   45.597   42.500
H   50.653   45.313   40.991
H   52.364   44.866   41.032
H   49.049   48.377   43.354
H   46.824   49.239   45.166
H   48.447   48.359   46.240
H   50.020   48.514   45.335
H   49.359   46.982   45.707
H   46.867   49.646   42.124
H   48.140   50.394   43.113
H   45.903   51.276   44.406
H   45.725   51.677   42.669
H   47.084   52.234   43.430
H   41.446   48.896   43.161
H   41.565   48.961   44.888
H   40.297   47.999   44.094
H   40.444   42.924   42.821
H   39.377   42.718   47.472
H   41.010   42.921   48.109
H   40.573   41.516   47.159
H   41.726   41.688   36.643
H   43.377   40.806   36.477
H   42.748   43.439   35.392
H   45.575   40.906   36.165
H   46.569   41.853   35.036
H   45.372   40.714   34.392
H   45.866   43.980   34.015
H   44.246   44.455   33.906
H   44.572   41.977   32.332
H   45.755   43.179   31.916
H   43.301   44.596   31.885
H   42.994   42.905   31.280
H   45.352   43.612   29.958
H   44.265   46.064   29.021
H   45.643   45.840   30.041
H   44.078   46.192   30.700
H   42.656   44.009   28.649
H   43.237   42.443   29.163
H   45.092   42.408   27.775
H   45.297   44.184   27.781
H   42.941   43.398   26.338
H   44.292   42.578   25.658
H   45.439   44.914   25.837
H   43.250   46.536   24.944
H   43.989   46.538   26.605
H   42.591   45.538   26.190
H   43.688   44.893   23.429
H   44.426   43.432   23.769
H   46.072   45.901   23.857
H   45.424   45.411   22.297
H   46.621   43.233   22.498
H   46.974   43.487   24.219
H   48.007   45.375   22.067
H   48.755   42.497   21.925
H   50.176   43.603   22.126
H   49.106   43.884   20.853
H   49.751   45.827   23.346
H   49.816   44.352   24.295
H   48.580   45.606   24.644
