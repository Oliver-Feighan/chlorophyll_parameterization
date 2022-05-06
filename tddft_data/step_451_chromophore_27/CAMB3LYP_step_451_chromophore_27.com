%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_451_chromophore_27 TDDFT with cam-b3lyp functional

0 1
Mg   -6.029   24.614   27.109
C   -4.398   26.639   29.645
C   -7.171   22.573   29.651
C   -7.196   22.757   24.713
C   -4.455   26.782   24.800
N   -5.843   24.714   29.443
C   -5.143   25.566   30.251
C   -5.342   25.217   31.688
C   -5.929   23.723   31.582
C   -6.341   23.655   30.134
C   -5.092   22.494   32.030
C   -6.221   26.190   32.562
C   -5.810   26.334   34.025
C   -5.976   25.099   34.988
O   -5.468   24.034   34.875
O   -6.856   25.350   36.024
N   -7.058   22.927   27.195
C   -7.442   22.224   28.321
C   -8.040   20.949   27.859
C   -8.109   20.987   26.397
C   -7.394   22.198   26.044
C   -8.532   19.789   28.804
C   -8.711   20.002   25.401
O   -8.403   20.147   24.246
C   -9.666   18.787   25.839
N   -5.904   24.803   25.096
C   -6.432   23.812   24.321
C   -6.349   24.257   22.860
C   -5.544   25.548   22.853
C   -5.303   25.744   24.401
C   -5.805   23.179   21.912
C   -6.226   26.778   22.268
C   -5.317   27.672   21.404
N   -4.704   26.390   27.168
C   -4.191   27.128   26.125
C   -3.312   28.194   26.661
C   -3.413   28.074   28.047
C   -4.239   26.928   28.301
C   -2.714   29.323   25.924
C   -2.808   28.561   29.359
O   -1.983   29.446   29.598
C   -3.453   27.630   30.394
C   -2.351   27.004   31.091
O   -1.613   26.168   30.588
O   -2.307   27.512   32.393
C   -1.455   26.848   33.365
C   -7.158   24.311   37.006
C   -8.324   23.348   36.711
C   -8.913   22.494   37.547
C   -8.426   22.312   38.988
C   -10.141   21.638   37.054
C   -11.487   22.033   37.686
C   -11.942   20.961   38.602
C   -12.847   19.971   37.892
C   -14.290   20.446   38.168
C   -12.536   18.484   38.405
C   -11.318   17.892   37.737
C   -10.787   16.612   38.428
C   -9.234   16.626   38.555
C   -8.724   17.799   39.417
C   -8.691   15.296   39.076
C   -7.559   14.803   38.147
C   -6.107   15.427   38.454
C   -5.055   14.268   38.527
C   -5.111   13.499   39.875
C   -3.598   14.757   38.137
H   -7.484   21.820   30.377
H   -7.518   22.091   23.910
H   -4.153   27.400   23.951
H   -4.363   25.122   32.158
H   -6.821   23.681   32.207
H   -4.512   22.223   31.148
H   -5.794   21.723   32.348
H   -4.415   22.866   32.798
H   -7.193   25.698   32.605
H   -6.314   27.155   32.064
H   -6.484   27.140   34.315
H   -4.779   26.682   34.087
H   -9.618   19.868   28.842
H   -8.094   19.709   29.799
H   -8.247   18.870   28.290
H   -9.119   17.849   25.742
H   -10.603   18.737   25.284
H   -9.972   18.836   26.884
H   -7.354   24.418   22.470
H   -4.603   25.339   22.345
H   -6.011   22.173   22.277
H   -4.729   23.346   21.867
H   -6.207   23.229   20.900
H   -6.629   27.402   23.065
H   -7.127   26.408   21.779
H   -4.255   27.424   21.427
H   -5.470   28.693   21.753
H   -5.870   27.819   20.476
H   -3.370   29.784   25.186
H   -1.764   28.979   25.516
H   -2.577   30.236   26.504
H   -4.104   28.084   31.141
H   -0.470   27.268   33.572
H   -1.385   25.783   33.147
H   -1.898   26.886   34.360
H   -7.498   24.825   37.905
H   -6.278   23.688   37.168
H   -8.725   23.391   35.698
H   -7.593   22.982   39.201
H   -8.204   21.263   39.183
H   -9.173   22.709   39.676
H   -9.819   20.611   37.224
H   -10.369   21.767   35.996
H   -12.215   22.251   36.905
H   -11.533   22.970   38.242
H   -12.456   21.399   39.457
H   -11.036   20.501   38.996
H   -12.642   19.925   36.822
H   -14.824   19.745   38.809
H   -14.820   20.430   37.215
H   -14.414   21.471   38.517
H   -13.263   17.883   37.859
H   -12.523   18.213   39.461
H   -10.643   18.743   37.648
H   -11.605   17.706   36.702
H   -10.986   15.849   37.675
H   -11.299   16.196   39.295
H   -8.845   16.841   37.559
H   -9.640   18.187   39.865
H   -8.294   18.523   38.725
H   -7.991   17.546   40.183
H   -9.525   14.603   39.192
H   -8.362   15.413   40.109
H   -7.819   14.889   37.092
H   -7.493   13.741   38.384
H   -6.186   15.887   39.439
H   -5.841   16.168   37.700
H   -5.357   13.526   37.787
H   -4.120   13.480   40.329
H   -5.469   12.470   39.835
H   -5.817   14.061   40.486
H   -3.679   15.722   37.637
H   -3.202   13.964   37.504
H   -3.085   14.813   39.097

