%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1751_chromophore_4 TDDFT with cam-b3lyp functional

0 1
Mg   9.223   4.170   28.022
C   10.730   2.504   30.685
C   7.739   6.196   30.230
C   7.673   5.516   25.399
C   10.632   1.659   25.838
N   9.219   4.309   30.187
C   9.988   3.550   31.059
C   9.888   4.112   32.459
C   8.743   5.217   32.391
C   8.629   5.294   30.876
C   7.380   4.926   33.129
C   11.193   4.745   32.815
C   11.699   4.623   34.229
C   10.664   4.260   35.275
O   9.695   4.987   35.460
O   10.860   3.101   35.955
N   7.946   5.668   27.825
C   7.470   6.405   28.832
C   6.563   7.439   28.300
C   6.464   7.160   26.935
C   7.446   6.114   26.624
C   5.928   8.487   29.204
C   5.609   7.857   25.859
O   5.595   7.547   24.674
C   4.594   8.848   26.344
N   9.105   3.536   25.888
C   8.401   4.413   25.019
C   8.711   3.877   23.584
C   9.842   2.910   23.749
C   9.874   2.676   25.220
C   7.555   3.200   22.819
C   11.184   3.338   23.239
C   12.011   2.264   22.458
N   10.439   2.452   28.153
C   10.977   1.553   27.212
C   11.740   0.455   27.867
C   11.771   0.860   29.242
C   10.899   2.038   29.349
C   12.369   -0.701   27.241
C   12.234   0.519   30.545
O   13.029   -0.339   30.955
C   11.606   1.552   31.540
C   10.775   0.765   32.498
O   9.840   0.070   32.163
O   11.050   0.975   33.788
C   10.320   0.220   34.772
C   10.113   2.713   37.128
C   10.532   3.666   38.222
C   10.989   3.540   39.480
C   11.257   2.226   40.177
C   11.133   4.795   40.294
C   9.792   5.232   40.952
C   9.524   6.730   40.813
C   8.933   7.353   42.102
C   7.503   6.734   42.409
C   8.999   8.907   42.024
C   10.080   9.410   43.036
C   9.889   10.865   43.456
C   10.293   11.056   44.954
C   11.167   12.331   45.111
C   9.026   11.041   45.930
C   8.942   9.723   46.720
C   7.522   9.271   47.075
C   7.266   7.785   46.752
C   5.775   7.535   46.344
C   7.646   6.937   47.949
H   7.188   6.784   30.967
H   7.230   5.934   24.493
H   11.058   1.007   25.072
H   9.568   3.280   33.086
H   9.100   6.136   32.857
H   7.327   5.437   34.090
H   7.350   3.851   33.305
H   6.569   5.272   32.488
H   11.219   5.813   32.599
H   11.974   4.386   32.146
H   12.159   5.575   34.496
H   12.467   3.849   34.222
H   4.846   8.421   29.323
H   6.195   9.362   28.611
H   6.270   8.532   30.238
H   5.057   9.780   26.667
H   4.030   8.261   27.069
H   3.846   9.028   25.571
H   9.070   4.724   22.998
H   9.600   1.971   23.251
H   7.749   3.341   21.755
H   6.623   3.635   23.178
H   7.362   2.189   23.179
H   11.791   3.655   24.087
H   11.132   4.273   22.682
H   12.501   2.684   21.579
H   11.480   1.378   22.109
H   12.793   1.909   23.129
H   12.704   -0.624   26.207
H   11.633   -1.491   27.093
H   13.071   -1.252   27.868
H   12.430   2.023   32.076
H   10.974   0.090   35.634
H   9.970   -0.745   34.406
H   9.474   0.823   35.104
H   10.442   1.696   37.343
H   9.033   2.677   36.985
H   10.190   4.690   38.070
H   10.763   2.046   41.132
H   12.332   2.250   40.361
H   11.120   1.426   39.449
H   11.499   5.524   39.571
H   11.887   4.611   41.059
H   9.872   5.003   42.015
H   8.916   4.739   40.530
H   8.897   6.935   39.945
H   10.442   7.263   40.562
H   9.670   7.021   42.834
H   7.070   6.211   41.556
H   6.746   7.476   42.663
H   7.626   6.077   43.271
H   8.025   9.341   42.249
H   9.314   9.279   41.049
H   11.001   9.323   42.459
H   10.188   8.861   43.971
H   8.869   11.174   43.228
H   10.487   11.533   42.837
H   10.908   10.248   45.350
H   10.648   13.013   45.784
H   11.089   12.905   44.188
H   12.193   12.179   45.446
H   8.222   11.095   45.196
H   8.944   11.924   46.563
H   9.436   9.987   47.655
H   9.483   8.912   46.233
H   6.889   9.982   46.545
H   7.374   9.415   48.145
H   7.893   7.443   45.928
H   5.659   7.428   45.266
H   5.103   8.307   46.720
H   5.591   6.591   46.859
H   8.095   6.036   47.531
H   6.817   6.537   48.534
H   8.348   7.354   48.670

