%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1_chromophore_27 ZINDO

0 1
Mg   -5.210   24.541   27.195
C   -3.541   26.384   29.693
C   -6.380   22.499   29.726
C   -6.783   22.591   24.876
C   -3.732   26.393   24.817
N   -5.009   24.475   29.485
C   -4.393   25.403   30.316
C   -4.778   25.157   31.761
C   -5.380   23.705   31.740
C   -5.612   23.568   30.252
C   -4.487   22.583   32.401
C   -5.828   26.148   32.285
C   -5.660   26.572   33.748
C   -5.928   25.520   34.860
O   -5.178   24.808   35.420
O   -7.247   25.651   35.240
N   -6.480   22.853   27.281
C   -6.765   22.191   28.426
C   -7.522   21.026   28.101
C   -7.817   21.045   26.694
C   -6.985   22.142   26.232
C   -7.947   20.031   29.145
C   -8.630   20.100   25.809
O   -8.907   20.385   24.623
C   -9.087   18.724   26.280
N   -5.320   24.558   25.170
C   -5.973   23.615   24.382
C   -5.699   23.910   22.943
C   -4.772   25.174   22.837
C   -4.608   25.452   24.340
C   -5.249   22.679   22.100
C   -5.373   26.380   22.083
C   -4.944   26.457   20.605
N   -3.895   26.018   27.181
C   -3.381   26.779   26.132
C   -2.501   27.909   26.610
C   -2.550   27.715   28.000
C   -3.426   26.623   28.301
C   -1.787   28.935   25.749
C   -2.033   28.274   29.240
O   -1.249   29.202   29.443
C   -2.781   27.475   30.390
C   -1.665   26.984   31.333
O   -0.581   26.549   31.003
O   -2.033   27.276   32.645
C   -1.033   26.854   33.626
C   -7.564   24.944   36.511
C   -8.753   24.010   36.285
C   -9.206   23.034   37.178
C   -8.642   22.672   38.542
C   -10.439   22.214   36.802
C   -11.605   22.545   37.716
C   -12.141   21.385   38.532
C   -12.837   20.297   37.695
C   -14.376   20.604   37.604
C   -12.473   18.878   38.236
C   -11.075   18.359   37.854
C   -10.252   18.044   39.133
C   -8.796   17.661   38.813
C   -7.707   18.676   39.491
C   -8.488   16.152   39.028
C   -7.214   15.591   38.264
C   -6.276   14.932   39.298
C   -4.745   15.127   38.870
C   -3.893   13.923   39.358
C   -4.266   16.497   39.329
H   -6.796   21.923   30.555
H   -7.246   22.043   24.053
H   -3.201   26.863   23.987
H   -3.942   25.157   32.460
H   -6.324   23.596   32.274
H   -3.587   22.935   32.906
H   -4.090   21.819   31.733
H   -5.035   22.071   33.192
H   -6.837   25.751   32.176
H   -5.779   27.088   31.735
H   -6.234   27.469   33.980
H   -4.650   26.933   33.941
H   -7.011   19.854   29.675
H   -8.293   19.055   28.805
H   -8.678   20.404   29.862
H   -8.257   18.167   26.715
H   -9.457   18.282   25.355
H   -9.853   18.807   27.051
H   -6.669   24.060   22.468
H   -3.772   24.984   22.445
H   -5.161   21.829   22.777
H   -4.267   22.885   21.675
H   -6.057   22.545   21.380
H   -5.165   27.269   22.678
H   -6.454   26.277   21.986
H   -4.658   25.466   20.252
H   -4.057   27.051   20.384
H   -5.810   26.770   20.021
H   -2.547   29.614   25.362
H   -1.261   28.538   24.881
H   -1.105   29.457   26.420
H   -3.505   28.048   30.969
H   -1.160   25.838   34.001
H   -1.124   27.594   34.421
H   0.018   26.831   33.338
H   -7.981   25.702   37.173
H   -6.805   24.425   37.095
H   -9.302   24.101   35.348
H   -8.292   21.656   38.362
H   -9.496   22.631   39.219
H   -7.753   23.206   38.876
H   -10.150   21.164   36.741
H   -10.782   22.469   35.800
H   -12.410   22.862   37.052
H   -11.396   23.503   38.192
H   -12.859   21.829   39.222
H   -11.377   20.940   39.170
H   -12.531   20.202   36.653
H   -14.660   21.651   37.502
H   -14.864   20.149   38.466
H   -14.707   20.129   36.681
H   -13.228   18.159   37.915
H   -12.479   19.016   39.317
H   -10.479   19.056   37.265
H   -11.241   17.402   37.358
H   -10.608   17.170   39.678
H   -10.286   18.836   39.881
H   -8.582   17.670   37.744
H   -7.003   18.174   40.154
H   -8.335   19.350   40.074
H   -7.156   19.274   38.765
H   -9.352   15.546   38.758
H   -8.252   16.061   40.089
H   -6.637   16.369   37.763
H   -7.396   14.835   37.501
H   -6.461   13.864   39.412
H   -6.274   15.379   40.292
H   -4.618   15.064   37.789
H   -4.547   13.094   39.626
H   -3.447   14.255   40.295
H   -3.045   13.638   38.736
H   -3.370   16.434   39.947
H   -4.981   16.928   40.031
H   -3.993   17.069   38.442

