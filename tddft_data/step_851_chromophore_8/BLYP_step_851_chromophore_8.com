%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_851_chromophore_8 TDDFT with blyp functional

0 1
Mg   45.454   2.942   46.791
C   43.518   5.698   46.648
C   42.601   0.946   46.996
C   47.261   0.123   46.394
C   48.230   4.935   45.707
N   43.327   3.353   46.632
C   42.769   4.538   46.650
C   41.236   4.404   46.447
C   41.000   2.967   47.027
C   42.435   2.319   46.859
C   40.586   2.960   48.551
C   40.723   4.501   44.899
C   39.944   5.745   44.449
C   40.266   6.353   43.063
O   40.696   7.475   42.886
O   39.933   5.466   42.063
N   45.045   0.792   46.941
C   43.819   0.217   46.935
C   43.992   -1.184   46.995
C   45.346   -1.517   46.813
C   45.984   -0.181   46.735
C   42.785   -2.052   47.099
C   45.834   -2.938   46.686
O   45.063   -3.883   46.779
C   47.313   -3.221   46.634
N   47.431   2.594   46.161
C   47.941   1.320   46.169
C   49.399   1.303   45.628
C   49.751   2.865   45.421
C   48.396   3.512   45.715
C   50.294   0.583   46.604
C   50.367   3.266   44.076
C   51.799   3.925   44.004
N   45.827   4.884   46.285
C   46.983   5.541   45.993
C   46.703   6.939   46.240
C   45.343   7.039   46.506
C   44.880   5.769   46.515
C   47.552   8.076   46.077
C   44.282   8.023   46.662
O   44.248   9.231   46.743
C   43.034   7.127   46.823
C   42.327   7.379   48.086
O   42.700   6.994   49.193
O   41.114   7.912   47.820
C   40.293   8.191   49.005
C   40.082   6.001   40.669
C   40.928   5.079   39.926
C   41.362   5.328   38.683
C   41.046   6.616   37.778
C   42.393   4.413   38.054
C   41.996   3.734   36.695
C   41.820   2.233   36.760
C   41.909   1.672   35.323
C   40.743   0.673   34.995
C   43.242   0.969   35.016
C   43.825   1.311   33.721
C   44.920   2.372   33.952
C   45.237   3.114   32.688
C   46.809   3.149   32.536
C   44.631   4.618   32.625
C   43.430   4.669   31.715
C   43.791   4.810   30.227
C   43.231   3.765   29.213
C   41.675   3.885   29.188
C   43.821   3.885   27.818
H   41.607   0.497   47.034
H   47.962   -0.714   46.403
H   49.092   5.591   45.565
H   40.640   5.067   47.073
H   40.214   2.459   46.468
H   41.266   2.507   49.273
H   39.641   2.424   48.632
H   40.580   4.001   48.875
H   40.244   3.574   44.584
H   41.656   4.550   44.337
H   39.893   6.516   45.218
H   38.917   5.407   44.305
H   42.889   -2.955   46.497
H   41.983   -1.488   46.623
H   42.448   -2.214   48.123
H   48.007   -2.879   47.402
H   47.618   -2.995   45.613
H   47.332   -4.309   46.697
H   49.435   0.807   44.658
H   50.464   3.231   46.159
H   49.885   0.544   47.613
H   51.250   1.100   46.686
H   50.531   -0.421   46.254
H   49.603   3.664   43.408
H   50.404   2.358   43.475
H   52.295   4.118   44.956
H   51.680   4.856   43.450
H   52.320   3.188   43.394
H   48.141   8.019   45.162
H   48.251   8.179   46.906
H   46.916   8.957   45.990
H   42.353   7.429   46.028
H   39.423   8.792   48.743
H   40.838   8.656   49.827
H   39.886   7.214   49.267
H   40.196   7.083   40.609
H   39.151   5.886   40.114
H   41.375   4.247   40.469
H   42.009   6.995   37.439
H   40.509   7.450   38.229
H   40.521   6.422   36.842
H   42.604   3.646   38.799
H   43.263   5.059   37.939
H   42.792   3.841   35.958
H   41.036   4.175   36.426
H   40.842   2.085   37.219
H   42.654   1.816   37.325
H   41.753   2.560   34.709
H   41.054   -0.352   34.795
H   40.203   1.156   34.180
H   40.089   0.669   35.867
H   43.165   -0.119   35.016
H   43.907   1.184   35.853
H   43.072   1.846   33.143
H   44.237   0.418   33.252
H   45.855   1.955   34.327
H   44.523   3.125   34.633
H   44.810   2.581   31.839
H   47.382   2.223   32.590
H   47.236   3.911   33.187
H   47.146   3.499   31.560
H   45.452   5.177   32.175
H   44.439   5.095   33.586
H   43.001   5.649   31.922
H   42.729   3.847   31.863
H   44.867   4.651   30.154
H   43.596   5.826   29.881
H   43.576   2.767   29.484
H   41.349   4.086   28.168
H   41.305   4.626   29.896
H   41.293   2.902   29.465
H   44.662   3.192   27.781
H   44.079   4.943   27.776
H   43.160   3.631   26.990

