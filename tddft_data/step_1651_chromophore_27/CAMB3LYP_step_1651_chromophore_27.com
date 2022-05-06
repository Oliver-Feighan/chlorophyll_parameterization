%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1651_chromophore_27 TDDFT with cam-b3lyp functional

0 1
Mg   -5.023   25.154   26.315
C   -3.448   27.161   28.744
C   -5.873   22.989   28.973
C   -6.081   22.801   24.166
C   -3.701   27.118   23.766
N   -4.672   25.060   28.586
C   -4.174   26.132   29.357
C   -4.298   25.967   30.858
C   -4.969   24.486   30.890
C   -5.161   24.107   29.433
C   -3.927   23.491   31.531
C   -5.282   26.980   31.588
C   -5.240   26.941   33.127
C   -6.532   26.440   33.789
O   -7.591   27.000   33.766
O   -6.225   25.399   34.623
N   -6.035   23.199   26.515
C   -6.270   22.546   27.696
C   -6.977   21.329   27.407
C   -7.082   21.238   25.996
C   -6.419   22.472   25.487
C   -7.442   20.394   28.476
C   -7.772   20.200   25.087
O   -7.931   20.390   23.894
C   -8.465   19.014   25.675
N   -4.985   24.999   24.295
C   -5.478   23.908   23.606
C   -5.145   24.092   22.084
C   -4.519   25.514   21.992
C   -4.431   25.960   23.415
C   -4.418   22.877   21.392
C   -5.333   26.490   21.103
C   -4.550   26.960   19.824
N   -3.874   26.904   26.247
C   -3.366   27.538   25.143
C   -2.569   28.724   25.595
C   -2.614   28.573   27.030
C   -3.370   27.430   27.385
C   -1.851   29.629   24.722
C   -2.076   29.056   28.268
O   -1.264   29.998   28.477
C   -2.791   28.322   29.430
C   -1.694   27.837   30.454
O   -0.618   27.308   30.275
O   -2.142   28.257   31.689
C   -1.296   27.867   32.781
C   -6.991   25.236   35.856
C   -8.008   24.132   35.680
C   -8.406   23.105   36.509
C   -7.969   22.871   37.906
C   -9.541   22.116   36.073
C   -10.836   22.259   36.820
C   -11.317   21.203   37.774
C   -12.547   20.417   37.333
C   -13.835   21.135   37.807
C   -12.496   18.976   37.789
C   -11.776   17.969   36.962
C   -11.040   16.880   37.839
C   -9.544   17.096   38.029
C   -9.166   18.248   38.956
C   -8.852   15.759   38.526
C   -7.350   15.665   38.141
C   -6.431   15.323   39.306
C   -5.090   14.583   39.011
C   -4.652   14.088   40.419
C   -4.069   15.551   38.374
H   -6.114   22.323   29.805
H   -6.399   22.061   23.428
H   -3.194   27.671   22.972
H   -3.333   25.997   31.364
H   -5.922   24.465   31.419
H   -3.142   24.051   32.039
H   -3.460   22.881   30.757
H   -4.417   22.861   32.274
H   -6.344   26.826   31.397
H   -5.124   28.009   31.264
H   -5.078   27.977   33.426
H   -4.365   26.359   33.416
H   -6.823   19.507   28.606
H   -8.311   19.832   28.133
H   -7.675   20.778   29.470
H   -9.222   19.400   26.358
H   -7.770   18.404   26.252
H   -8.950   18.341   24.969
H   -6.175   24.100   21.729
H   -3.505   25.338   21.634
H   -4.562   22.034   22.068
H   -3.337   23.015   21.354
H   -4.885   22.623   20.441
H   -5.663   27.351   21.684
H   -6.256   25.966   20.859
H   -5.162   26.992   18.923
H   -3.665   26.327   19.765
H   -4.256   27.978   20.081
H   -2.081   29.585   23.658
H   -0.792   29.422   24.874
H   -1.891   30.666   25.054
H   -3.400   29.132   29.832
H   -0.756   28.758   33.103
H   -0.649   26.992   32.733
H   -2.026   27.664   33.564
H   -7.662   26.033   36.175
H   -6.393   24.909   36.706
H   -8.449   23.994   34.693
H   -7.416   21.932   37.912
H   -8.721   22.668   38.668
H   -7.152   23.528   38.205
H   -9.084   21.140   36.234
H   -9.762   22.212   35.010
H   -11.581   22.347   36.029
H   -10.738   23.135   37.461
H   -11.583   21.663   38.726
H   -10.528   20.469   37.933
H   -12.474   20.493   36.248
H   -14.164   21.849   37.052
H   -13.596   21.671   38.726
H   -14.671   20.448   37.934
H   -13.501   18.555   37.808
H   -12.064   18.964   38.790
H   -10.989   18.565   36.501
H   -12.432   17.603   36.173
H   -11.231   15.940   37.320
H   -11.352   16.659   38.859
H   -9.294   17.260   36.981
H   -8.870   17.829   39.918
H   -10.063   18.803   39.233
H   -8.342   18.836   38.551
H   -9.402   14.958   38.030
H   -8.971   15.665   39.605
H   -7.081   16.706   37.963
H   -7.159   14.871   37.419
H   -6.959   14.714   40.040
H   -6.213   16.285   39.770
H   -5.268   13.735   38.350
H   -4.728   13.013   40.256
H   -5.261   14.462   41.243
H   -3.628   14.289   40.731
H   -4.504   16.267   37.676
H   -3.283   14.964   37.899
H   -3.649   16.235   39.111

