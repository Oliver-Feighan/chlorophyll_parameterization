%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_701_chromophore_23 TDDFT with PBE1PBE functional

0 1
Mg   -11.007   40.418   42.528
C   -9.157   37.505   41.458
C   -8.701   42.338   40.758
C   -12.981   42.956   42.986
C   -13.588   38.084   43.472
N   -9.120   39.965   41.250
C   -8.551   38.736   41.001
C   -7.214   38.820   40.196
C   -7.002   40.349   40.208
C   -8.349   40.963   40.811
C   -5.738   40.768   40.821
C   -7.330   38.186   38.787
C   -6.778   36.814   38.533
C   -5.792   36.608   37.409
O   -4.718   37.248   37.303
O   -6.132   35.568   36.571
N   -10.848   42.424   42.035
C   -9.814   43.075   41.292
C   -10.101   44.505   41.190
C   -11.301   44.704   41.960
C   -11.803   43.318   42.311
C   -9.192   45.435   40.519
C   -11.955   46.036   42.244
O   -11.531   47.086   41.819
C   -13.180   46.198   42.990
N   -12.937   40.486   43.179
C   -13.554   41.697   43.327
C   -15.053   41.637   43.781
C   -15.235   40.044   43.786
C   -13.885   39.464   43.450
C   -15.330   42.290   45.151
C   -16.279   39.412   42.787
C   -17.594   38.909   43.387
N   -11.313   38.321   42.551
C   -12.389   37.541   42.989
C   -11.941   36.175   43.093
C   -10.729   36.113   42.421
C   -10.410   37.471   42.081
C   -12.764   35.041   43.645
C   -9.694   35.179   42.037
O   -9.616   33.939   42.057
C   -8.626   36.072   41.483
C   -7.378   35.998   42.302
O   -7.026   36.642   43.297
O   -6.515   35.140   41.675
C   -5.344   34.554   42.448
C   -5.286   35.358   35.420
C   -5.976   35.569   34.080
C   -6.032   36.608   33.281
C   -5.369   37.957   33.673
C   -6.734   36.589   31.950
C   -5.803   36.372   30.757
C   -6.328   35.615   29.604
C   -5.855   36.342   28.278
C   -5.277   35.221   27.395
C   -7.055   37.038   27.561
C   -7.580   38.268   28.283
C   -9.091   38.526   28.149
C   -9.715   39.158   29.425
C   -10.425   38.173   30.345
C   -10.716   40.291   28.937
C   -9.930   41.651   28.818
C   -9.656   42.335   30.172
C   -8.170   42.452   30.586
C   -7.933   41.987   32.075
C   -7.829   44.002   30.541
H   -7.915   42.935   40.289
H   -13.718   43.758   42.909
H   -14.414   37.467   43.831
H   -6.426   38.423   40.837
H   -6.839   40.758   39.211
H   -5.461   41.822   40.792
H   -4.900   40.314   40.292
H   -5.715   40.441   41.861
H   -6.849   38.853   38.071
H   -8.374   38.228   38.478
H   -7.632   36.165   38.339
H   -6.393   36.506   39.505
H   -9.019   46.254   41.217
H   -9.604   45.741   39.557
H   -8.221   45.007   40.270
H   -13.089   45.584   43.886
H   -13.990   45.825   42.363
H   -13.332   47.255   43.209
H   -15.805   42.133   43.167
H   -15.348   39.760   44.833
H   -14.460   42.299   45.807
H   -16.123   41.811   45.725
H   -15.619   43.335   45.042
H   -15.781   38.633   42.209
H   -16.544   40.289   42.197
H   -17.913   39.642   44.128
H   -17.255   38.007   43.896
H   -18.355   38.542   42.698
H   -12.068   34.240   43.893
H   -13.444   34.666   42.880
H   -13.369   35.309   44.512
H   -8.399   35.668   40.497
H   -4.554   34.156   41.812
H   -5.780   33.664   42.900
H   -4.930   35.215   43.210
H   -4.800   34.390   35.299
H   -4.470   36.074   35.330
H   -6.480   34.686   33.687
H   -4.402   37.947   34.174
H   -6.082   38.373   34.385
H   -5.355   38.700   32.876
H   -7.163   37.591   31.929
H   -7.502   35.825   32.070
H   -4.931   35.911   31.221
H   -5.418   37.353   30.477
H   -7.412   35.658   29.504
H   -5.948   34.601   29.726
H   -5.098   37.124   28.335
H   -4.188   35.277   27.373
H   -5.599   35.265   26.355
H   -5.645   34.244   27.709
H   -7.792   36.269   27.328
H   -6.595   37.330   26.617
H   -7.067   39.022   27.687
H   -7.291   38.250   29.334
H   -9.427   37.499   28.003
H   -9.246   39.014   27.186
H   -9.012   39.646   30.100
H   -11.476   38.160   30.059
H   -10.275   38.618   31.329
H   -10.093   37.136   30.296
H   -11.594   40.495   29.551
H   -11.035   39.983   27.941
H   -10.404   42.349   28.127
H   -8.985   41.497   28.297
H   -10.179   41.951   31.048
H   -10.071   43.343   30.178
H   -7.546   41.848   29.927
H   -7.834   40.904   32.141
H   -8.720   42.282   32.770
H   -7.024   42.459   32.447
H   -8.452   44.666   29.943
H   -6.874   44.033   30.017
H   -7.622   44.443   31.516

