%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_151_chromophore_25 TDDFT with PBE1PBE functional

0 1
Mg   -3.219   34.533   27.465
C   -4.020   32.829   30.332
C   -1.322   36.676   29.324
C   -2.570   36.397   24.566
C   -4.981   32.345   25.544
N   -2.661   34.645   29.614
C   -3.138   33.867   30.604
C   -2.658   34.227   32.032
C   -1.894   35.560   31.709
C   -1.957   35.713   30.148
C   -2.454   36.820   32.397
C   -1.802   33.203   32.792
C   -2.108   33.048   34.272
C   -0.957   33.091   35.287
O   0.153   32.660   35.168
O   -1.541   33.631   36.414
N   -2.044   36.337   27.010
C   -1.384   37.032   27.968
C   -0.789   38.192   27.326
C   -1.300   38.167   25.928
C   -1.976   36.907   25.736
C   0.053   39.239   28.058
C   -1.186   39.191   24.806
O   -1.759   39.066   23.772
C   -0.464   40.407   25.129
N   -3.782   34.496   25.455
C   -3.356   35.321   24.410
C   -3.818   34.845   23.007
C   -4.473   33.506   23.399
C   -4.345   33.407   24.903
C   -4.868   35.819   22.370
C   -3.863   32.313   22.772
C   -2.415   32.013   23.021
N   -4.367   32.930   27.733
C   -5.022   32.143   26.899
C   -5.602   31.077   27.628
C   -5.350   31.345   29.041
C   -4.526   32.432   29.025
C   -6.381   30.004   27.078
C   -5.625   30.955   30.388
O   -6.233   30.046   30.902
C   -4.792   31.951   31.291
C   -5.746   32.635   32.156
O   -6.794   33.120   31.748
O   -5.279   32.603   33.442
C   -6.354   32.988   34.416
C   -0.726   33.251   37.585
C   -1.569   33.627   38.759
C   -1.600   34.811   39.359
C   -0.975   36.082   38.687
C   -2.517   35.100   40.472
C   -1.947   34.643   41.847
C   -1.644   35.802   42.903
C   -1.622   35.324   44.387
C   -3.014   35.206   45.020
C   -0.852   36.365   45.272
C   0.216   35.736   46.208
C   1.649   36.304   45.956
C   2.485   35.277   45.152
C   3.964   35.515   45.492
C   2.332   35.498   43.624
C   2.511   34.292   42.689
C   3.723   34.337   41.694
C   3.286   34.541   40.232
C   4.567   34.301   39.398
C   2.677   35.884   39.965
H   -0.573   37.275   29.846
H   -2.367   36.952   23.647
H   -5.484   31.616   24.906
H   -3.523   34.437   32.661
H   -0.845   35.572   32.004
H   -3.382   37.003   31.854
H   -1.797   37.682   32.281
H   -2.581   36.679   33.470
H   -0.781   33.580   32.731
H   -1.964   32.273   32.247
H   -2.421   32.007   34.349
H   -2.929   33.672   34.625
H   -0.364   40.240   27.940
H   1.036   39.294   27.591
H   0.141   39.033   29.125
H   0.431   40.031   25.625
H   -1.081   41.006   25.799
H   -0.148   40.837   24.178
H   -3.042   34.668   22.262
H   -5.492   33.509   23.014
H   -5.217   36.548   23.102
H   -5.687   35.353   21.823
H   -4.415   36.394   21.563
H   -3.907   32.394   21.686
H   -4.374   31.381   23.014
H   -1.943   32.890   23.464
H   -1.955   31.761   22.065
H   -2.373   31.119   23.642
H   -5.806   29.509   26.295
H   -7.344   30.194   26.605
H   -6.622   29.260   27.838
H   -4.131   31.305   31.868
H   -6.941   33.805   33.997
H   -5.945   33.293   35.379
H   -7.020   32.138   34.562
H   0.223   33.764   37.743
H   -0.659   32.166   37.669
H   -2.155   32.872   39.283
H   -0.193   36.492   39.327
H   -1.680   36.859   38.394
H   -0.346   35.861   37.824
H   -3.445   34.589   40.214
H   -2.805   36.151   40.498
H   -0.990   34.135   41.733
H   -2.620   33.952   42.354
H   -2.283   36.682   42.834
H   -0.715   36.311   42.644
H   -1.171   34.337   44.487
H   -2.976   35.000   46.090
H   -3.416   34.350   44.477
H   -3.647   36.060   44.781
H   -1.549   36.991   45.828
H   -0.446   37.134   44.615
H   0.283   34.660   46.046
H   0.058   36.028   47.247
H   2.114   36.448   46.931
H   1.675   37.233   45.386
H   2.230   34.249   45.409
H   4.181   34.851   46.329
H   4.130   36.488   45.954
H   4.692   35.224   44.735
H   2.909   36.392   43.386
H   1.265   35.634   43.450
H   1.575   34.093   42.167
H   2.513   33.406   43.324
H   4.334   33.475   41.961
H   4.277   35.226   41.996
H   2.559   33.759   40.012
H   5.506   34.262   39.950
H   4.857   35.111   38.729
H   4.415   33.355   38.879
H   3.077   36.656   40.622
H   1.656   35.572   40.185
H   2.795   36.239   38.941

