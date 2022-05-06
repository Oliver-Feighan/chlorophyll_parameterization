%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_101_chromophore_5 TDDFT with blyp functional

0 1
Mg   24.029   -6.865   45.475
C   26.159   -4.664   43.864
C   21.414   -5.478   43.416
C   22.232   -9.735   45.824
C   26.778   -8.627   46.806
N   23.866   -5.180   43.814
C   24.907   -4.301   43.484
C   24.472   -3.195   42.476
C   22.822   -3.320   42.692
C   22.656   -4.756   43.287
C   22.254   -2.167   43.575
C   24.945   -3.453   40.982
C   25.499   -2.216   40.184
C   24.945   -1.893   38.791
O   23.875   -1.323   38.597
O   25.696   -2.521   37.800
N   22.011   -7.415   44.898
C   21.049   -6.689   44.103
C   19.833   -7.448   44.037
C   20.074   -8.685   44.714
C   21.497   -8.691   45.159
C   18.588   -6.969   43.378
C   18.927   -9.677   44.869
O   17.802   -9.414   44.444
C   19.016   -10.957   45.740
N   24.481   -9.036   46.002
C   23.519   -9.878   46.314
C   24.073   -11.172   46.916
C   25.601   -10.815   47.358
C   25.643   -9.457   46.598
C   23.239   -11.916   48.029
C   26.672   -11.801   46.856
C   27.454   -12.621   47.936
N   26.014   -6.626   45.545
C   27.015   -7.356   46.122
C   28.235   -6.642   45.939
C   28.008   -5.656   44.999
C   26.599   -5.718   44.717
C   29.580   -6.901   46.615
C   28.610   -4.555   44.264
O   29.769   -4.195   44.324
C   27.432   -3.993   43.293
C   27.387   -2.555   43.412
O   27.620   -1.774   42.464
O   26.881   -2.149   44.639
C   26.891   -0.677   44.833
C   25.063   -2.737   36.520
C   24.279   -4.060   36.496
C   24.330   -5.114   35.647
C   25.249   -5.308   34.454
C   23.462   -6.363   35.917
C   22.031   -6.338   35.468
C   21.950   -7.110   34.102
C   21.010   -6.439   33.073
C   19.673   -7.271   32.864
C   21.595   -6.115   31.656
C   22.110   -4.660   31.453
C   23.570   -4.647   30.984
C   23.707   -3.754   29.723
C   25.097   -3.060   29.701
C   23.392   -4.445   28.305
C   22.457   -3.542   27.379
C   23.176   -2.939   26.136
C   22.705   -3.500   24.710
C   23.930   -3.784   23.793
C   21.776   -2.506   23.963
H   20.706   -4.851   42.869
H   21.693   -10.680   45.911
H   27.639   -9.010   47.358
H   24.798   -2.264   42.939
H   22.343   -3.171   41.724
H   21.538   -1.569   43.011
H   22.984   -1.519   44.060
H   21.688   -2.450   44.462
H   24.037   -3.813   40.498
H   25.732   -4.203   40.899
H   26.578   -2.335   40.082
H   25.439   -1.384   40.885
H   18.217   -7.561   42.541
H   18.764   -5.970   42.977
H   17.707   -6.882   44.014
H   19.336   -10.569   46.707
H   19.634   -11.688   45.218
H   18.038   -11.414   45.893
H   24.021   -11.882   46.090
H   25.505   -10.575   48.417
H   22.351   -11.310   48.213
H   23.773   -11.845   48.977
H   22.989   -12.957   47.825
H   27.451   -11.230   46.352
H   26.162   -12.401   46.102
H   27.002   -12.454   48.914
H   28.529   -12.460   47.849
H   27.269   -13.692   47.856
H   29.410   -7.197   47.650
H   30.159   -5.985   46.730
H   30.165   -7.663   46.098
H   27.537   -4.266   42.243
H   27.900   -0.282   44.709
H   26.614   -0.483   45.870
H   26.276   -0.163   44.094
H   25.789   -2.551   35.729
H   24.288   -2.000   36.309
H   23.539   -4.173   37.289
H   25.824   -4.436   34.143
H   24.795   -5.589   33.504
H   25.927   -6.153   34.574
H   23.489   -6.607   36.979
H   23.980   -7.194   35.437
H   21.658   -5.321   35.346
H   21.446   -6.839   36.239
H   21.575   -8.075   34.445
H   22.911   -7.202   33.597
H   20.703   -5.487   33.505
H   18.935   -7.210   33.663
H   20.009   -8.295   33.029
H   19.233   -7.223   31.868
H   20.935   -6.183   30.791
H   22.314   -6.916   31.486
H   22.054   -4.014   32.330
H   21.539   -4.207   30.643
H   23.988   -5.647   30.868
H   24.084   -4.203   31.837
H   23.047   -2.891   29.811
H   25.790   -3.610   29.065
H   25.467   -2.948   30.720
H   24.947   -2.037   29.355
H   22.872   -5.355   28.606
H   24.273   -4.770   27.751
H   22.064   -2.713   27.967
H   21.618   -4.145   27.031
H   24.247   -3.044   26.312
H   22.983   -1.866   26.156
H   22.086   -4.370   24.929
H   24.014   -3.032   23.008
H   23.804   -4.692   23.204
H   24.844   -3.920   24.370
H   21.584   -1.685   24.654
H   20.743   -2.813   23.795
H   22.250   -2.029   23.105

