%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_801_chromophore_1 TDDFT with PBE1PBE functional

0 1
Mg   -1.837   17.149   26.398
C   -1.865   14.920   29.003
C   -2.942   19.634   28.516
C   -1.906   19.215   23.689
C   -1.239   14.410   24.141
N   -2.527   17.207   28.411
C   -2.271   16.267   29.332
C   -2.620   16.735   30.760
C   -3.349   18.121   30.490
C   -2.892   18.359   29.105
C   -4.851   18.060   30.649
C   -1.395   16.803   31.768
C   -1.606   17.677   33.043
C   -1.270   16.985   34.408
O   -1.882   16.090   34.952
O   -0.317   17.665   35.066
N   -2.292   19.074   26.199
C   -2.795   19.962   27.161
C   -2.861   21.314   26.630
C   -2.439   21.177   25.251
C   -2.203   19.805   24.967
C   -3.284   22.471   27.474
C   -2.317   22.308   24.170
O   -2.180   22.079   22.943
C   -2.332   23.803   24.545
N   -1.649   16.835   24.302
C   -1.751   17.836   23.395
C   -1.317   17.346   22.034
C   -0.810   15.955   22.245
C   -1.281   15.689   23.685
C   -2.410   17.556   20.905
C   0.663   15.591   22.000
C   1.655   16.350   22.737
N   -1.620   15.064   26.458
C   -1.384   14.128   25.468
C   -1.254   12.841   26.159
C   -1.373   13.116   27.561
C   -1.664   14.471   27.673
C   -1.039   11.513   25.545
C   -1.252   12.554   28.828
O   -1.035   11.423   29.264
C   -1.793   13.614   29.790
C   -1.015   13.554   31.130
O   0.129   13.984   31.235
O   -1.606   12.710   32.033
C   -0.797   12.336   33.222
C   -0.015   17.307   36.424
C   -0.567   18.357   37.344
C   -0.692   18.142   38.660
C   -0.127   17.041   39.635
C   -1.431   19.288   39.387
C   -0.722   20.721   39.373
C   -0.778   21.405   40.770
C   0.424   22.264   41.091
C   0.727   23.282   39.984
C   1.678   21.475   41.646
C   2.747   22.440   42.095
C   3.777   21.802   43.099
C   3.632   22.476   44.516
C   4.972   22.416   45.176
C   2.541   21.761   45.338
C   1.335   22.678   45.688
C   -0.003   21.936   45.606
C   -0.959   22.596   46.597
C   -0.568   22.258   48.046
C   -2.399   22.131   46.275
H   -3.391   20.432   29.112
H   -1.736   19.814   22.792
H   -0.909   13.666   23.413
H   -3.280   15.960   31.149
H   -3.006   18.884   31.188
H   -5.374   18.948   31.004
H   -5.138   17.419   31.482
H   -5.225   17.585   29.742
H   -0.588   17.236   31.178
H   -1.134   15.810   32.136
H   -2.552   18.217   33.040
H   -0.847   18.459   33.040
H   -4.022   22.294   28.257
H   -3.740   23.290   26.918
H   -2.383   22.824   27.976
H   -1.927   23.882   25.554
H   -3.388   24.071   24.588
H   -1.804   24.381   23.787
H   -0.380   17.822   21.744
H   -1.328   15.226   21.621
H   -2.131   18.433   20.320
H   -3.427   17.722   21.260
H   -2.522   16.755   20.174
H   0.804   16.005   21.001
H   0.720   14.504   21.957
H   1.130   16.754   23.602
H   2.001   17.173   22.111
H   2.465   15.716   23.097
H   -0.912   11.646   24.470
H   -2.023   11.061   25.663
H   -0.377   10.797   26.034
H   -2.771   13.182   30.000
H   0.207   11.978   32.995
H   -1.453   11.647   33.755
H   -0.565   13.255   33.759
H   1.066   17.437   36.490
H   -0.277   16.310   36.779
H   -0.972   19.259   36.885
H   0.189   16.173   39.057
H   -0.959   16.715   40.260
H   0.691   17.466   40.217
H   -1.402   19.018   40.442
H   -2.419   19.346   38.930
H   -1.260   21.430   38.743
H   0.321   20.583   39.089
H   -0.919   20.619   41.512
H   -1.712   21.966   40.791
H   0.159   22.714   42.048
H   0.837   24.285   40.397
H   -0.021   23.370   39.196
H   1.678   23.096   39.486
H   2.147   20.912   40.839
H   1.533   20.693   42.391
H   2.333   23.428   42.298
H   3.344   22.856   41.283
H   4.755   21.793   42.619
H   3.514   20.753   43.231
H   3.305   23.509   44.394
H   4.895   22.572   46.252
H   5.605   23.271   44.937
H   5.496   21.479   44.985
H   3.006   21.512   46.292
H   2.210   20.825   44.889
H   1.277   23.449   44.920
H   1.410   23.264   46.604
H   0.201   20.908   45.903
H   -0.440   22.000   44.610
H   -0.864   23.669   46.435
H   -1.307   21.859   48.741
H   -0.253   23.177   48.540
H   0.256   21.546   48.091
H   -2.550   21.561   45.358
H   -3.205   22.860   46.187
H   -2.724   21.439   47.051

