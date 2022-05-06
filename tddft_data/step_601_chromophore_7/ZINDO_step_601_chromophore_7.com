%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_601_chromophore_7 ZINDO

0 1
Mg   26.043   -0.900   29.285
C   28.002   -0.891   32.258
C   23.263   -0.585   31.298
C   24.207   -0.661   26.587
C   28.904   -1.164   27.457
N   25.698   -0.845   31.538
C   26.643   -0.768   32.522
C   26.020   -0.535   33.908
C   24.493   -0.461   33.597
C   24.455   -0.590   32.066
C   23.680   -1.577   34.212
C   26.543   0.776   34.596
C   25.944   1.170   35.920
C   26.895   1.304   37.138
O   28.064   0.943   37.113
O   26.230   1.614   38.278
N   24.029   -0.650   28.988
C   23.035   -0.688   29.925
C   21.766   -0.667   29.348
C   21.996   -0.470   27.906
C   23.462   -0.648   27.723
C   20.529   -0.563   30.210
C   20.982   -0.327   26.795
O   21.307   -0.095   25.583
C   19.488   -0.299   27.116
N   26.455   -1.170   27.261
C   25.548   -0.845   26.344
C   26.180   -0.717   24.930
C   27.673   -0.738   25.258
C   27.680   -1.096   26.764
C   25.715   -1.926   24.030
C   28.462   0.558   24.907
C   29.841   0.378   24.155
N   28.054   -1.116   29.720
C   29.104   -1.286   28.862
C   30.361   -1.303   29.600
C   30.001   -1.208   30.935
C   28.563   -1.070   30.928
C   31.711   -1.383   28.965
C   30.515   -1.195   32.337
O   31.654   -1.529   32.734
C   29.240   -0.894   33.229
C   29.080   -1.848   34.375
O   28.629   -2.953   34.281
O   29.526   -1.252   35.557
C   29.413   -2.125   36.727
C   26.878   1.239   39.512
C   26.024   1.506   40.631
C   26.408   1.857   41.886
C   27.839   1.734   42.356
C   25.362   2.231   42.908
C   24.800   1.116   43.775
C   23.287   1.092   43.911
C   22.821   1.005   45.475
C   21.497   0.372   45.708
C   22.850   2.431   46.080
C   23.553   2.452   47.455
C   22.687   3.180   48.480
C   23.150   4.735   48.530
C   23.555   5.183   49.975
C   21.989   5.595   48.086
C   22.330   6.880   47.295
C   23.072   6.794   46.045
C   22.129   6.775   44.744
C   22.656   5.684   43.792
C   22.122   8.203   44.151
H   22.405   -0.479   31.966
H   23.795   -0.686   25.576
H   29.807   -1.158   26.844
H   26.412   -1.291   34.588
H   24.095   0.493   33.942
H   24.178   -2.298   34.860
H   23.118   -2.181   33.500
H   22.984   -1.263   34.990
H   26.465   1.625   33.916
H   27.585   0.573   34.843
H   25.042   0.591   36.120
H   25.456   2.136   35.785
H   19.825   -1.316   29.857
H   20.208   0.466   30.045
H   20.474   -0.729   31.286
H   18.962   0.229   26.321
H   19.251   0.267   28.017
H   19.187   -1.331   27.297
H   25.827   0.255   24.586
H   28.191   -1.506   24.684
H   26.586   -2.416   23.595
H   25.045   -1.639   23.219
H   25.201   -2.702   24.596
H   28.724   1.127   25.799
H   27.878   1.157   24.208
H   29.707   0.330   23.074
H   30.186   -0.634   24.369
H   30.651   1.068   24.389
H   31.782   -2.327   28.426
H   32.377   -1.346   29.827
H   31.918   -0.616   28.218
H   29.361   0.129   33.584
H   30.120   -1.782   37.482
H   29.619   -3.172   36.505
H   28.387   -1.926   37.036
H   27.792   1.810   39.675
H   27.012   0.160   39.436
H   24.977   1.741   40.440
H   28.469   1.155   41.680
H   27.876   1.341   43.372
H   28.364   2.684   42.456
H   24.545   2.727   42.384
H   25.842   3.046   43.451
H   25.246   1.075   44.769
H   25.038   0.137   43.359
H   22.958   0.173   43.426
H   22.711   1.916   43.488
H   23.533   0.356   45.985
H   21.562   -0.715   45.769
H   20.892   0.587   44.828
H   20.997   0.721   46.612
H   21.842   2.833   46.178
H   23.412   3.153   45.487
H   24.499   2.987   47.378
H   23.796   1.468   47.855
H   23.031   2.767   49.428
H   21.629   2.998   48.291
H   23.915   4.899   47.771
H   24.638   5.106   50.068
H   23.031   4.757   50.831
H   23.296   6.240   50.043
H   21.387   5.864   48.954
H   21.311   4.965   47.510
H   22.876   7.551   47.958
H   21.423   7.445   47.081
H   23.835   6.016   46.018
H   23.665   7.707   45.984
H   21.075   6.602   44.961
H   22.203   4.751   44.126
H   23.729   5.506   43.871
H   22.360   5.818   42.752
H   21.127   8.551   43.876
H   22.677   8.175   43.214
H   22.596   8.980   44.751

