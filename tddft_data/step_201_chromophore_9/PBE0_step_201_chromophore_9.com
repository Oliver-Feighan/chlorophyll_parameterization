%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_201_chromophore_9 TDDFT with PBE1PBE functional

0 1
Mg   35.810   1.355   30.040
C   33.748   2.071   32.662
C   38.448   1.481   32.330
C   38.032   0.841   27.499
C   33.221   1.638   27.760
N   36.087   1.739   32.263
C   35.018   1.905   33.161
C   35.492   1.925   34.618
C   37.054   2.076   34.467
C   37.227   1.733   32.877
C   37.668   3.487   34.826
C   35.158   0.534   35.315
C   34.876   0.631   36.907
C   35.217   1.976   37.755
O   34.379   2.677   38.375
O   36.572   2.174   37.827
N   37.954   1.134   29.915
C   38.843   1.153   30.969
C   40.166   0.840   30.510
C   40.117   0.667   29.080
C   38.683   0.804   28.755
C   41.372   0.666   31.368
C   41.162   0.489   28.037
O   40.962   0.263   26.856
C   42.604   0.682   28.441
N   35.643   1.068   27.970
C   36.642   1.024   27.104
C   36.165   0.934   25.642
C   34.613   1.168   25.766
C   34.435   1.239   27.325
C   36.801   1.940   24.598
C   33.658   0.073   25.148
C   33.520   -1.283   25.933
N   33.879   1.772   30.138
C   32.918   1.866   29.157
C   31.630   2.186   29.754
C   31.879   2.211   31.203
C   33.238   1.938   31.337
C   30.305   2.367   28.988
C   31.307   2.315   32.535
O   30.129   2.443   32.871
C   32.512   2.197   33.574
C   32.327   3.312   34.512
O   32.763   4.510   34.292
O   31.730   2.881   35.692
C   31.313   3.967   36.606
C   37.048   3.130   38.861
C   37.806   2.235   39.839
C   37.456   1.727   41.066
C   36.118   1.790   41.819
C   38.336   0.646   41.831
C   39.626   1.136   42.501
C   40.240   0.205   43.584
C   41.737   -0.298   43.040
C   41.859   -1.823   43.383
C   42.789   0.554   43.780
C   43.749   1.032   42.799
C   45.160   1.308   43.310
C   45.724   2.657   42.748
C   47.281   2.736   42.648
C   45.220   3.912   43.657
C   44.627   4.948   42.653
C   45.314   6.260   42.431
C   44.329   7.369   42.173
C   43.847   7.364   40.668
C   44.855   8.778   42.654
H   39.319   1.520   32.987
H   38.694   0.761   26.634
H   32.495   1.778   26.957
H   35.050   2.780   35.129
H   37.542   1.356   35.124
H   36.877   4.237   34.805
H   38.416   3.784   34.091
H   38.058   3.394   35.839
H   35.828   -0.256   34.978
H   34.197   0.410   34.815
H   35.287   -0.263   37.377
H   33.802   0.449   36.958
H   41.122   0.816   32.418
H   42.209   1.268   31.014
H   41.662   -0.380   31.264
H   42.766   -0.017   29.262
H   42.649   1.732   28.730
H   43.377   0.522   27.690
H   36.337   -0.128   25.468
H   34.425   2.153   25.338
H   37.440   1.443   23.868
H   37.280   2.805   25.057
H   36.036   2.448   24.011
H   34.088   -0.152   24.172
H   32.643   0.433   24.979
H   34.056   -2.016   25.330
H   32.492   -1.534   26.192
H   34.003   -1.231   26.909
H   30.523   2.337   27.920
H   29.839   3.294   29.322
H   29.662   1.562   29.344
H   32.302   1.285   34.133
H   32.151   4.549   36.988
H   30.664   3.568   37.386
H   30.804   4.771   36.074
H   36.233   3.712   39.291
H   37.738   3.855   38.429
H   38.770   1.878   39.477
H   35.917   0.864   42.357
H   35.314   2.009   41.116
H   36.175   2.590   42.557
H   38.615   -0.063   41.052
H   37.834   0.098   42.628
H   39.389   2.083   42.985
H   40.311   1.341   41.678
H   39.565   -0.611   43.843
H   40.177   0.843   44.466
H   41.802   -0.103   41.970
H   41.250   -2.351   42.650
H   41.502   -2.031   44.392
H   42.860   -2.217   43.208
H   43.172   0.002   44.639
H   42.325   1.461   44.169
H   43.357   1.929   42.318
H   44.055   0.290   42.062
H   45.876   0.542   43.014
H   45.179   1.434   44.392
H   45.501   2.613   41.682
H   47.547   3.751   42.945
H   47.607   2.467   41.644
H   47.731   1.979   43.290
H   45.998   4.467   44.180
H   44.451   3.632   44.377
H   43.573   4.954   42.930
H   44.705   4.502   41.662
H   46.032   6.198   41.614
H   45.890   6.576   43.301
H   43.377   7.161   42.662
H   42.863   6.906   40.572
H   44.480   6.797   39.984
H   43.870   8.363   40.233
H   45.461   8.784   43.560
H   43.925   9.277   42.926
H   45.434   9.188   41.826

