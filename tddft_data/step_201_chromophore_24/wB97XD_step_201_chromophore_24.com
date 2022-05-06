%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_201_chromophore_24 TDDFT with wB97XD functional

0 1
Mg   -0.157   43.853   24.711
C   2.064   43.700   27.430
C   -2.696   42.783   26.829
C   -2.169   44.068   22.188
C   2.725   44.277   22.667
N   -0.304   43.293   26.935
C   0.695   43.436   27.840
C   0.249   43.019   29.264
C   -1.236   42.504   28.956
C   -1.476   42.954   27.463
C   -1.514   41.021   29.242
C   0.260   44.154   30.251
C   0.140   43.769   31.759
C   0.118   42.322   32.148
O   1.082   41.550   32.295
O   -1.171   41.840   32.309
N   -2.224   43.596   24.543
C   -3.023   43.112   25.528
C   -4.350   43.130   24.941
C   -4.242   43.462   23.573
C   -2.873   43.784   23.352
C   -5.533   42.679   25.664
C   -5.287   43.512   22.443
O   -5.114   43.982   21.345
C   -6.667   42.888   22.699
N   0.294   44.097   22.749
C   -0.792   44.190   21.918
C   -0.244   44.137   20.422
C   1.203   44.467   20.653
C   1.498   44.309   22.088
C   -0.558   42.667   19.741
C   1.610   45.901   20.068
C   1.441   47.053   21.052
N   1.992   43.865   24.880
C   3.020   44.081   24.027
C   4.216   44.182   24.643
C   3.945   44.008   26.023
C   2.555   43.809   26.140
C   5.591   44.450   23.983
C   4.490   44.005   27.363
O   5.609   44.144   27.689
C   3.305   43.794   28.359
C   3.587   42.504   29.165
O   3.809   41.376   28.683
O   3.851   42.843   30.430
C   4.432   41.770   31.202
C   -1.230   40.464   32.925
C   -2.347   40.423   33.988
C   -2.200   40.587   35.312
C   -0.987   41.161   36.024
C   -3.365   40.302   36.258
C   -4.545   41.405   36.211
C   -5.949   40.617   36.044
C   -7.055   41.335   36.877
C   -7.309   42.839   36.435
C   -8.280   40.370   36.773
C   -9.107   40.228   38.017
C   -10.561   39.857   37.802
C   -11.337   39.872   39.194
C   -11.460   38.417   39.765
C   -12.761   40.519   39.016
C   -12.942   41.867   39.720
C   -12.496   43.023   38.770
C   -11.406   43.916   39.388
C   -11.753   45.408   39.370
C   -9.999   43.589   38.831
H   -3.589   42.382   27.313
H   -2.787   44.185   21.295
H   3.589   44.538   22.053
H   0.849   42.158   29.558
H   -1.988   43.027   29.547
H   -2.082   40.588   28.419
H   -2.202   41.007   30.087
H   -0.625   40.512   29.615
H   -0.528   44.849   29.964
H   1.165   44.715   30.018
H   -0.662   44.410   32.126
H   1.035   44.136   32.262
H   -5.524   42.875   26.736
H   -5.714   41.638   25.398
H   -6.380   43.194   25.209
H   -6.507   41.987   23.291
H   -7.008   42.703   21.680
H   -7.388   43.478   23.264
H   -0.826   44.797   19.778
H   1.697   43.632   20.156
H   -1.191   42.056   20.385
H   0.295   41.996   19.643
H   -1.192   42.778   18.862
H   0.982   46.003   19.183
H   2.644   45.952   19.726
H   1.101   46.680   22.018
H   0.597   47.605   20.640
H   2.372   47.617   20.997
H   5.759   43.630   23.284
H   6.402   44.439   24.711
H   5.654   45.422   23.494
H   3.164   44.655   29.012
H   4.570   42.200   32.193
H   5.435   41.596   30.812
H   3.782   40.896   31.171
H   -0.319   40.161   33.440
H   -1.403   39.736   32.132
H   -3.330   39.994   33.795
H   -0.146   41.341   35.354
H   -0.561   40.489   36.769
H   -1.327   42.041   36.571
H   -3.063   40.097   37.285
H   -3.745   39.388   35.801
H   -4.575   42.008   35.303
H   -4.601   42.154   37.001
H   -5.873   39.589   36.398
H   -6.303   40.548   35.015
H   -6.583   41.431   37.854
H   -6.715   43.075   35.552
H   -7.083   43.566   37.215
H   -8.375   42.901   36.218
H   -7.969   39.342   36.585
H   -8.885   40.648   35.910
H   -9.072   41.148   38.601
H   -8.567   39.472   38.586
H   -10.577   38.847   37.393
H   -10.893   40.635   37.114
H   -10.732   40.388   39.940
H   -11.475   37.576   39.073
H   -12.389   38.404   40.335
H   -10.714   38.300   40.552
H   -13.463   39.812   39.459
H   -13.174   40.545   38.007
H   -12.628   41.861   40.764
H   -14.017   42.033   39.804
H   -13.363   43.683   38.744
H   -12.093   42.723   37.802
H   -11.417   43.576   40.424
H   -12.087   45.703   40.365
H   -12.551   45.703   38.689
H   -10.879   46.001   39.099
H   -9.398   44.470   38.609
H   -10.054   43.042   37.890
H   -9.455   43.024   39.589

