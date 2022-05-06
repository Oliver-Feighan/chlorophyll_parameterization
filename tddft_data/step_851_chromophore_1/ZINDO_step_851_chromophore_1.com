%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_851_chromophore_1 ZINDO

0 1
Mg   -1.304   18.078   26.611
C   -1.519   15.709   29.338
C   -1.997   20.536   28.823
C   -1.669   20.151   23.919
C   -1.035   15.397   24.442
N   -1.661   17.996   28.849
C   -1.796   17.017   29.730
C   -2.195   17.498   31.123
C   -2.363   19.063   30.974
C   -2.016   19.220   29.442
C   -3.790   19.506   31.276
C   -1.208   17.090   32.328
C   -0.956   18.099   33.506
C   -0.057   17.618   34.671
O   1.014   17.005   34.644
O   -0.692   17.824   35.874
N   -1.658   20.095   26.437
C   -1.936   20.945   27.454
C   -2.126   22.238   26.923
C   -1.903   22.201   25.514
C   -1.806   20.744   25.280
C   -2.442   23.414   27.830
C   -1.721   23.280   24.564
O   -1.664   23.102   23.380
C   -1.828   24.709   25.016
N   -1.523   17.812   24.460
C   -1.449   18.836   23.609
C   -1.596   18.347   22.185
C   -0.934   16.909   22.319
C   -1.172   16.651   23.833
C   -3.051   18.330   21.680
C   0.542   16.737   21.997
C   1.499   17.863   22.504
N   -1.214   15.962   26.834
C   -1.198   14.981   25.809
C   -1.152   13.730   26.433
C   -1.254   14.023   27.819
C   -1.362   15.352   28.050
C   -1.119   12.359   25.758
C   -1.324   13.318   29.053
O   -1.185   12.135   29.366
C   -1.675   14.414   30.056
C   -0.771   14.266   31.256
O   0.370   14.670   31.387
O   -1.323   13.502   32.250
C   -0.674   13.004   33.467
C   0.050   17.296   37.095
C   -0.268   18.277   38.308
C   0.116   18.082   39.602
C   0.852   16.807   40.096
C   -0.237   19.138   40.587
C   0.940   19.975   41.129
C   0.710   21.445   41.199
C   1.755   22.364   40.569
C   1.675   22.338   39.000
C   3.189   22.069   40.943
C   3.764   22.942   42.120
C   3.791   22.167   43.453
C   4.502   22.889   44.544
C   6.128   23.000   44.296
C   4.011   22.452   45.954
C   2.792   23.355   46.351
C   1.518   22.573   46.620
C   0.268   23.117   45.935
C   -0.454   24.082   46.849
C   -0.639   22.005   45.254
H   -2.190   21.290   29.589
H   -1.973   20.840   23.128
H   -0.883   14.683   23.630
H   -3.182   17.103   31.364
H   -1.685   19.700   31.542
H   -3.687   20.306   32.009
H   -4.409   18.723   31.715
H   -4.330   19.931   30.431
H   -0.234   16.932   31.865
H   -1.646   16.215   32.809
H   -1.793   18.562   34.029
H   -0.424   18.946   33.072
H   -2.130   23.190   28.850
H   -3.462   23.692   27.565
H   -1.746   24.242   27.694
H   -2.645   24.831   25.729
H   -1.796   25.477   24.244
H   -1.026   24.823   25.746
H   -0.984   19.014   21.578
H   -1.373   16.229   21.590
H   -3.319   19.290   21.238
H   -3.815   18.123   22.429
H   -3.203   17.530   20.955
H   0.733   16.521   20.945
H   0.990   15.880   22.499
H   1.635   18.482   21.618
H   2.469   17.386   22.644
H   1.254   18.346   23.450
H   -2.096   12.248   25.286
H   -1.033   11.625   26.558
H   -0.329   12.337   25.008
H   -2.711   14.264   30.361
H   -0.516   11.926   33.501
H   -1.386   13.252   34.254
H   0.271   13.507   33.677
H   1.125   17.413   36.964
H   -0.309   16.269   37.166
H   -0.733   19.238   38.088
H   0.157   16.282   40.751
H   1.830   16.979   40.547
H   1.145   16.131   39.293
H   -0.602   18.597   41.460
H   -1.063   19.769   40.256
H   1.678   19.777   40.352
H   1.307   19.547   42.061
H   0.665   21.749   42.245
H   -0.262   21.698   40.776
H   1.531   23.400   40.824
H   2.512   21.938   38.427
H   1.645   23.394   38.729
H   0.707   21.934   38.706
H   3.862   22.272   40.110
H   3.364   21.031   41.226
H   3.004   23.689   42.349
H   4.747   23.398   42.005
H   4.287   21.227   43.211
H   2.772   21.965   43.785
H   4.335   23.958   44.417
H   6.359   22.378   43.432
H   6.612   22.682   45.219
H   6.361   24.048   44.104
H   4.837   22.490   46.664
H   3.855   21.373   45.934
H   2.569   24.123   45.611
H   3.165   24.096   47.059
H   1.356   22.597   47.698
H   1.548   21.520   46.341
H   0.434   23.703   45.031
H   -1.078   23.368   47.388
H   -0.987   24.817   46.247
H   0.195   24.546   47.592
H   -1.676   22.275   45.455
H   -0.467   21.026   45.703
H   -0.432   22.044   44.185

