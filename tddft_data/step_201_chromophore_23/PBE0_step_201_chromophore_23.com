%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_201_chromophore_23 TDDFT with PBE1PBE functional

0 1
Mg   -8.461   40.866   42.812
C   -7.589   37.591   42.042
C   -5.784   41.945   40.792
C   -9.695   43.869   43.046
C   -11.675   39.590   43.987
N   -6.907   39.884   41.557
C   -6.755   38.537   41.485
C   -5.604   38.235   40.572
C   -4.919   39.592   40.250
C   -5.925   40.540   40.889
C   -3.462   39.760   40.827
C   -6.027   37.487   39.275
C   -5.150   36.312   38.920
C   -4.292   36.637   37.727
O   -3.213   37.202   37.706
O   -4.923   36.206   36.570
N   -7.651   42.662   42.188
C   -6.699   42.923   41.295
C   -6.634   44.379   41.077
C   -7.688   44.996   41.780
C   -8.431   43.822   42.378
C   -5.421   44.969   40.369
C   -7.953   46.523   41.729
O   -7.297   47.304   41.064
C   -9.097   47.106   42.456
N   -10.383   41.603   43.494
C   -10.556   42.948   43.538
C   -11.914   43.365   44.096
C   -12.538   42.001   44.515
C   -11.470   40.975   43.940
C   -11.893   44.375   45.230
C   -14.003   41.785   44.045
C   -15.101   41.309   45.026
N   -9.442   38.999   43.034
C   -10.680   38.631   43.569
C   -10.782   37.180   43.610
C   -9.555   36.775   43.063
C   -8.776   37.903   42.717
C   -11.968   36.387   44.027
C   -8.919   35.576   42.642
O   -9.258   34.394   42.674
C   -7.601   36.097   41.916
C   -6.472   35.466   42.744
O   -6.438   35.531   43.949
O   -5.614   34.747   41.894
C   -4.407   34.118   42.493
C   -4.130   36.251   35.350
C   -4.867   35.791   34.083
C   -5.903   36.370   33.415
C   -6.420   37.740   33.707
C   -6.548   35.747   32.145
C   -5.694   35.804   30.864
C   -6.273   36.624   29.774
C   -5.474   37.930   29.587
C   -4.371   37.730   28.519
C   -6.431   39.102   29.213
C   -5.803   40.515   29.397
C   -6.498   41.479   30.349
C   -5.541   42.421   31.193
C   -5.439   42.001   32.680
C   -5.974   43.883   31.154
C   -5.717   44.558   29.823
C   -4.426   45.345   29.936
C   -4.657   46.878   29.624
C   -4.170   47.213   28.212
C   -3.981   47.729   30.731
H   -4.896   42.258   40.240
H   -10.125   44.871   43.109
H   -12.634   39.217   44.350
H   -4.872   37.622   41.098
H   -4.887   39.698   39.165
H   -2.733   39.517   40.053
H   -3.233   38.921   41.483
H   -3.293   40.674   41.396
H   -6.071   38.190   38.443
H   -7.087   37.246   39.350
H   -5.707   35.388   38.768
H   -4.460   36.048   39.721
H   -4.532   44.524   40.815
H   -5.251   46.035   40.524
H   -5.400   44.749   39.302
H   -9.057   48.193   42.387
H   -8.891   46.838   43.492
H   -10.019   46.758   41.990
H   -12.479   43.886   43.324
H   -12.519   41.841   45.593
H   -12.161   43.963   46.203
H   -12.564   45.222   45.089
H   -10.905   44.833   45.256
H   -14.091   41.246   43.102
H   -14.438   42.737   43.741
H   -14.809   41.224   46.073
H   -15.507   40.388   44.608
H   -16.040   41.861   44.977
H   -11.615   35.499   44.551
H   -12.381   36.119   43.054
H   -12.658   36.917   44.684
H   -7.474   35.740   40.894
H   -4.707   33.501   43.341
H   -3.839   34.917   42.971
H   -3.923   33.504   41.734
H   -3.353   35.495   35.471
H   -3.620   37.195   35.161
H   -4.438   34.882   33.663
H   -7.503   37.648   33.798
H   -6.276   38.431   32.877
H   -5.902   38.050   34.615
H   -7.549   36.132   31.952
H   -6.528   34.661   32.231
H   -5.609   34.799   30.450
H   -4.642   36.030   31.043
H   -7.351   36.745   29.880
H   -6.341   36.110   28.815
H   -5.123   38.099   30.605
H   -4.643   37.015   27.743
H   -3.404   37.462   28.944
H   -4.332   38.710   28.044
H   -7.295   38.952   29.861
H   -6.898   39.033   28.231
H   -5.978   41.084   28.484
H   -4.725   40.540   29.554
H   -7.257   41.032   30.990
H   -7.081   42.137   29.703
H   -4.506   42.321   30.866
H   -4.400   42.176   32.957
H   -5.691   40.952   32.832
H   -6.048   42.665   33.293
H   -5.567   44.372   32.040
H   -7.062   43.925   31.207
H   -6.561   45.118   29.421
H   -5.526   43.882   28.990
H   -3.811   44.787   29.230
H   -4.018   45.198   30.935
H   -5.713   47.145   29.612
H   -5.054   47.233   27.575
H   -3.509   46.389   27.944
H   -3.695   48.191   28.134
H   -4.817   48.238   31.211
H   -3.286   48.394   30.218
H   -3.466   47.173   31.513

