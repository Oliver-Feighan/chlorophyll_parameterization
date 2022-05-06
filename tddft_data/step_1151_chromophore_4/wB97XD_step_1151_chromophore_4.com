%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1151_chromophore_4 TDDFT with wB97XD functional

0 1
Mg   8.910   2.918   28.152
C   10.249   1.439   30.835
C   7.452   5.367   30.148
C   7.651   4.487   25.184
C   10.098   0.293   25.983
N   8.678   3.237   30.346
C   9.484   2.587   31.195
C   9.475   3.391   32.514
C   8.354   4.505   32.333
C   8.121   4.386   30.866
C   7.098   4.327   33.271
C   10.890   3.836   33.043
C   10.806   4.713   34.399
C   11.103   3.988   35.707
O   10.373   3.131   36.224
O   12.263   4.459   36.363
N   7.727   4.561   27.706
C   7.232   5.444   28.694
C   6.592   6.625   28.029
C   6.606   6.239   26.635
C   7.340   5.057   26.477
C   6.003   7.784   28.794
C   5.967   7.025   25.455
O   6.094   6.724   24.265
C   5.238   8.293   25.752
N   8.574   2.321   25.815
C   8.140   3.212   24.876
C   8.488   2.682   23.465
C   9.369   1.459   23.784
C   9.352   1.267   25.275
C   7.270   2.436   22.600
C   10.823   1.532   23.236
C   11.445   0.240   22.588
N   9.788   1.168   28.371
C   10.337   0.280   27.387
C   11.190   -0.695   28.055
C   11.095   -0.317   29.404
C   10.299   0.867   29.515
C   12.227   -1.620   27.485
C   11.647   -0.505   30.725
O   12.462   -1.393   31.051
C   11.069   0.555   31.698
C   10.246   -0.124   32.705
O   9.043   -0.418   32.674
O   11.029   -0.390   33.822
C   10.358   -1.062   34.919
C   12.595   3.800   37.675
C   12.329   4.754   38.871
C   12.377   4.549   40.217
C   12.890   3.238   40.897
C   11.959   5.669   41.140
C   10.463   5.604   41.612
C   9.585   6.783   41.210
C   9.455   7.904   42.321
C   8.030   7.952   42.847
C   9.897   9.288   41.845
C   11.245   9.775   42.396
C   11.269   11.273   42.707
C   11.720   11.469   44.229
C   12.679   12.724   44.276
C   10.522   11.768   45.183
C   10.886   11.748   46.704
C   9.616   11.171   47.412
C   9.647   9.609   47.575
C   8.287   8.950   47.857
C   10.536   9.264   48.707
H   7.055   6.182   30.757
H   7.388   5.031   24.275
H   10.622   -0.447   25.375
H   9.025   2.702   33.230
H   8.782   5.488   32.534
H   7.371   3.876   34.225
H   6.266   3.942   32.682
H   6.862   5.357   33.540
H   11.420   4.434   32.301
H   11.486   2.955   33.281
H   9.808   5.123   34.556
H   11.459   5.565   34.211
H   6.520   8.667   28.421
H   5.960   7.764   29.883
H   4.936   7.894   28.599
H   5.911   8.917   26.340
H   4.362   8.025   26.343
H   4.976   8.837   24.844
H   9.222   3.345   23.007
H   8.889   0.527   23.487
H   6.887   3.401   22.270
H   6.497   1.897   23.148
H   7.537   1.931   21.671
H   11.435   1.720   24.117
H   10.960   2.380   22.565
H   11.977   0.477   21.667
H   10.765   -0.592   22.408
H   12.115   -0.147   23.356
H   11.739   -2.467   27.003
H   13.039   -1.945   28.135
H   12.643   -1.022   26.674
H   11.938   0.993   32.190
H   11.024   -1.523   35.649
H   9.660   -1.778   34.484
H   9.804   -0.213   35.319
H   13.673   3.675   37.582
H   12.026   2.907   37.931
H   12.056   5.791   38.680
H   13.293   2.494   40.210
H   12.082   2.834   41.507
H   13.681   3.613   41.546
H   12.309   6.620   40.738
H   12.481   5.503   42.083
H   10.340   5.408   42.677
H   9.927   4.745   41.208
H   8.620   6.438   40.840
H   10.122   7.164   40.341
H   10.083   7.657   43.178
H   7.573   8.746   42.256
H   7.998   8.474   43.803
H   7.411   7.057   42.912
H   9.159   10.026   42.160
H   9.899   9.336   40.756
H   12.033   9.603   41.663
H   11.495   9.184   43.277
H   10.272   11.672   42.520
H   11.948   11.786   42.026
H   12.255   10.542   44.436
H   12.450   13.401   43.452
H   13.681   12.420   43.972
H   12.740   13.241   45.233
H   9.534   11.368   44.956
H   10.352   12.844   45.175
H   11.151   12.742   47.066
H   11.789   11.165   46.880
H   8.706   11.359   46.842
H   9.582   11.679   48.376
H   10.077   9.212   46.655
H   8.031   8.968   48.917
H   8.263   7.924   47.492
H   7.544   9.541   47.322
H   10.512   9.959   49.545
H   11.552   9.355   48.323
H   10.431   8.227   49.024

