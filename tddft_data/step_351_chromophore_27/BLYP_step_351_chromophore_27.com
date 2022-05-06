%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_351_chromophore_27 TDDFT with blyp functional

0 1
Mg   -5.473   24.873   26.908
C   -3.773   26.663   29.387
C   -6.456   22.758   29.448
C   -7.182   22.897   24.481
C   -4.139   26.660   24.410
N   -5.079   24.758   29.201
C   -4.469   25.671   29.974
C   -4.793   25.428   31.457
C   -5.308   23.941   31.447
C   -5.682   23.803   29.990
C   -4.328   22.792   31.885
C   -6.041   26.272   31.939
C   -6.496   26.262   33.431
C   -5.647   25.459   34.443
O   -4.423   25.425   34.583
O   -6.448   24.629   35.146
N   -6.695   23.155   26.961
C   -6.957   22.454   28.116
C   -7.629   21.219   27.651
C   -7.892   21.305   26.279
C   -7.277   22.539   25.848
C   -7.821   20.067   28.529
C   -8.697   20.323   25.479
O   -8.803   20.435   24.241
C   -9.393   19.053   26.024
N   -5.658   24.783   24.826
C   -6.357   23.940   23.960
C   -6.066   24.094   22.471
C   -5.158   25.430   22.582
C   -5.035   25.700   24.067
C   -5.583   22.875   21.692
C   -5.789   26.662   21.748
C   -4.883   27.319   20.651
N   -4.102   26.268   26.828
C   -3.699   27.005   25.690
C   -2.796   28.090   26.137
C   -2.843   28.040   27.612
C   -3.590   26.859   27.936
C   -2.186   29.109   25.321
C   -2.295   28.621   28.817
O   -1.611   29.588   28.970
C   -2.915   27.751   30.029
C   -1.681   27.268   30.784
O   -0.658   26.832   30.222
O   -1.750   27.428   32.148
C   -0.564   27.381   33.032
C   -5.731   23.741   36.049
C   -6.614   22.714   36.561
C   -7.686   22.836   37.391
C   -8.138   24.096   38.204
C   -8.512   21.550   37.662
C   -9.893   21.619   37.110
C   -10.912   21.315   38.183
C   -12.178   20.615   37.538
C   -13.532   21.439   37.711
C   -12.348   19.198   38.235
C   -11.395   18.199   37.615
C   -10.727   17.275   38.669
C   -9.277   16.983   38.403
C   -8.318   17.909   39.209
C   -9.001   15.540   38.615
C   -7.895   14.897   37.790
C   -6.414   15.229   38.337
C   -5.365   14.057   38.300
C   -5.296   13.356   39.668
C   -4.024   14.500   37.775
H   -6.743   22.058   30.236
H   -7.822   22.251   23.877
H   -3.764   27.298   23.606
H   -3.931   25.528   32.117
H   -6.243   23.763   31.979
H   -3.706   23.233   32.664
H   -3.903   22.258   31.035
H   -5.028   22.125   32.389
H   -6.847   26.006   31.255
H   -5.820   27.317   31.725
H   -7.475   25.788   33.377
H   -6.582   27.297   33.762
H   -8.863   19.817   28.728
H   -7.319   20.064   29.497
H   -7.378   19.214   28.015
H   -8.720   18.313   26.457
H   -10.049   18.639   25.258
H   -9.942   19.540   26.829
H   -7.076   24.333   22.139
H   -4.126   25.256   22.276
H   -5.015   22.090   22.192
H   -4.851   23.227   20.965
H   -6.353   22.449   21.049
H   -6.156   27.453   22.402
H   -6.709   26.323   21.271
H   -5.194   27.098   19.630
H   -3.819   27.090   20.704
H   -4.968   28.394   20.808
H   -3.033   29.656   24.907
H   -1.592   28.762   24.475
H   -1.685   29.803   25.997
H   -3.446   28.497   30.621
H   -0.351   28.316   33.550
H   0.317   27.113   32.449
H   -0.726   26.741   33.899
H   -5.321   24.397   36.818
H   -5.007   23.151   35.487
H   -6.330   21.732   36.182
H   -7.505   24.965   38.025
H   -8.315   23.804   39.239
H   -9.111   24.304   37.758
H   -8.561   21.356   38.733
H   -8.023   20.682   37.221
H   -10.033   20.905   36.299
H   -10.158   22.597   36.709
H   -11.142   22.162   38.830
H   -10.412   20.530   38.751
H   -12.041   20.541   36.460
H   -13.342   22.217   36.972
H   -13.846   21.692   38.724
H   -14.346   20.824   37.329
H   -13.386   18.865   38.233
H   -12.211   19.270   39.314
H   -10.663   18.718   36.996
H   -12.023   17.557   36.998
H   -11.248   16.336   38.852
H   -10.953   17.709   39.643
H   -8.970   17.162   37.373
H   -7.462   18.125   38.569
H   -7.850   17.394   40.048
H   -8.866   18.750   39.633
H   -9.894   14.934   38.459
H   -8.624   15.428   39.632
H   -7.928   15.461   36.858
H   -7.889   13.869   37.429
H   -6.467   15.710   39.314
H   -6.066   16.025   37.679
H   -5.742   13.322   37.588
H   -5.892   13.786   40.473
H   -4.295   13.351   40.098
H   -5.518   12.296   39.537
H   -3.954   14.200   36.730
H   -3.238   14.006   38.347
H   -3.955   15.574   37.948

