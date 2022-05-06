%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1801_chromophore_8 TDDFT with blyp functional

0 1
Mg   44.037   3.313   47.566
C   42.158   6.289   47.434
C   41.036   1.553   47.629
C   45.853   0.568   47.328
C   46.925   5.303   47.000
N   41.891   3.904   47.357
C   41.342   5.160   47.491
C   39.781   5.113   47.469
C   39.532   3.550   47.636
C   40.926   2.964   47.706
C   38.633   3.081   48.770
C   39.059   5.693   46.312
C   39.621   5.142   44.984
C   40.050   6.322   43.998
O   40.258   7.489   44.289
O   40.412   5.817   42.765
N   43.494   1.275   47.458
C   42.192   0.756   47.631
C   42.310   -0.673   47.493
C   43.659   -1.025   47.255
C   44.425   0.251   47.333
C   41.130   -1.553   47.672
C   44.177   -2.420   46.986
O   43.393   -3.406   47.017
C   45.622   -2.606   46.659
N   46.038   3.022   47.339
C   46.569   1.760   47.273
C   48.137   1.703   47.073
C   48.414   3.211   46.907
C   47.016   3.914   47.116
C   48.837   0.963   48.231
C   49.126   3.493   45.481
C   50.633   3.770   45.510
N   44.510   5.336   47.374
C   45.701   5.983   47.115
C   45.477   7.386   47.171
C   44.123   7.542   47.284
C   43.538   6.285   47.372
C   46.538   8.419   46.961
C   43.040   8.548   47.310
O   43.064   9.779   47.343
C   41.721   7.742   47.528
C   41.117   8.153   48.877
O   41.629   8.006   49.998
O   39.795   8.542   48.660
C   39.011   8.690   49.856
C   40.953   6.822   41.807
C   41.509   5.939   40.714
C   41.667   6.252   39.399
C   41.275   7.489   38.764
C   42.358   5.237   38.668
C   41.394   4.123   38.263
C   41.931   2.696   38.390
C   41.878   1.917   37.026
C   41.427   0.483   37.389
C   43.214   1.914   36.232
C   43.330   2.962   35.156
C   44.499   3.962   35.278
C   45.453   3.946   34.055
C   46.802   4.488   34.377
C   44.788   4.603   32.839
C   44.946   3.647   31.627
C   44.413   4.361   30.392
C   45.300   4.255   29.113
C   44.838   3.309   28.057
C   45.436   5.613   28.528
H   40.105   0.988   47.714
H   46.496   -0.312   47.282
H   47.872   5.807   46.798
H   39.360   5.546   48.377
H   39.033   3.145   46.756
H   39.367   2.938   49.563
H   38.102   2.133   48.693
H   37.987   3.899   49.087
H   39.200   6.773   46.356
H   38.003   5.490   46.494
H   38.955   4.420   44.512
H   40.486   4.482   45.064
H   40.976   -2.145   46.770
H   40.167   -1.054   47.779
H   41.261   -2.281   48.473
H   46.214   -2.442   47.559
H   45.777   -1.823   45.917
H   45.740   -3.586   46.198
H   48.340   1.214   46.121
H   49.147   3.601   47.614
H   49.707   1.542   48.541
H   49.152   -0.044   47.957
H   48.149   0.821   49.065
H   48.650   4.397   45.101
H   48.965   2.679   44.774
H   51.053   3.303   46.401
H   50.855   4.837   45.468
H   51.021   3.226   44.649
H   47.388   8.258   47.624
H   45.995   9.294   47.318
H   46.911   8.482   45.939
H   41.057   7.990   46.701
H   37.980   8.964   49.632
H   39.335   9.407   50.610
H   38.992   7.725   50.361
H   41.751   7.367   42.311
H   40.292   7.580   41.386
H   41.818   4.957   41.072
H   42.139   8.142   38.883
H   40.457   7.977   39.293
H   41.016   7.408   37.708
H   43.252   4.926   39.207
H   42.709   5.673   37.733
H   41.082   4.288   37.231
H   40.419   4.216   38.742
H   41.392   2.120   39.142
H   42.980   2.789   38.673
H   41.120   2.381   36.395
H   40.347   0.391   37.506
H   41.862   0.179   38.341
H   41.801   -0.258   36.682
H   43.431   0.951   35.769
H   44.087   2.022   36.877
H   42.397   3.524   35.122
H   43.520   2.485   34.194
H   45.127   3.920   36.168
H   44.061   4.953   35.395
H   45.550   2.889   33.806
H   47.260   5.107   33.606
H   47.431   3.601   34.457
H   46.950   5.075   35.283
H   45.217   5.591   32.672
H   43.744   4.652   33.151
H   44.346   2.737   31.664
H   45.919   3.192   31.446
H   44.071   5.358   30.670
H   43.488   3.820   30.195
H   46.342   4.054   29.361
H   43.832   3.576   27.735
H   44.691   2.249   28.262
H   45.551   3.352   27.234
H   44.572   6.240   28.748
H   45.456   5.601   27.438
H   46.317   6.082   28.965

