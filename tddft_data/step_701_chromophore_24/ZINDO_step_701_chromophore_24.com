%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_701_chromophore_24 ZINDO

0 1
Mg   -0.395   44.078   24.995
C   1.720   43.232   27.767
C   -3.009   42.868   26.912
C   -2.355   44.064   22.191
C   2.456   44.674   23.106
N   -0.549   43.038   27.147
C   0.367   43.026   28.139
C   -0.150   42.418   29.452
C   -1.698   42.142   29.052
C   -1.791   42.776   27.554
C   -2.170   40.678   29.080
C   -0.067   43.440   30.652
C   0.094   42.981   32.064
C   -0.544   41.651   32.477
O   0.032   40.527   32.580
O   -1.921   41.736   32.479
N   -2.468   43.858   24.599
C   -3.326   43.370   25.573
C   -4.690   43.454   25.037
C   -4.534   43.696   23.577
C   -3.081   43.824   23.407
C   -5.883   43.121   25.824
C   -5.622   43.817   22.462
O   -5.304   44.018   21.309
C   -7.045   43.844   22.948
N   -0.002   44.498   23.006
C   -0.966   44.393   22.013
C   -0.325   44.473   20.614
C   1.082   45.026   20.930
C   1.199   44.735   22.398
C   -0.338   43.185   19.872
C   1.336   46.514   20.682
C   0.638   47.529   21.654
N   1.661   44.045   25.349
C   2.673   44.366   24.457
C   3.874   44.247   25.147
C   3.558   43.850   26.416
C   2.205   43.653   26.543
C   5.178   44.510   24.742
C   4.120   43.552   27.727
O   5.291   43.466   28.053
C   2.946   43.179   28.631
C   3.194   41.886   29.241
O   3.297   40.876   28.577
O   3.521   41.904   30.597
C   3.946   40.689   31.221
C   -2.722   40.598   32.764
C   -3.647   40.898   33.954
C   -3.537   40.548   35.274
C   -2.268   39.921   35.856
C   -4.504   40.904   36.342
C   -4.441   42.346   36.859
C   -5.514   43.302   36.270
C   -6.609   43.739   37.316
C   -6.054   44.652   38.410
C   -7.833   44.351   36.581
C   -9.125   43.523   36.812
C   -9.265   42.295   35.740
C   -9.541   40.864   36.353
C   -9.071   39.707   35.405
C   -11.027   40.716   36.654
C   -11.495   41.143   38.048
C   -11.756   40.020   38.994
C   -12.984   39.217   38.643
C   -14.271   39.896   39.117
C   -12.932   37.811   39.257
H   -3.859   42.488   27.483
H   -2.876   44.109   21.232
H   3.309   44.882   22.457
H   0.332   41.464   29.668
H   -2.270   42.673   29.813
H   -1.294   40.044   29.212
H   -2.746   40.409   28.195
H   -2.752   40.751   29.998
H   -0.928   44.109   30.660
H   0.892   43.942   30.521
H   -0.081   43.816   32.743
H   1.168   42.833   32.175
H   -5.648   43.178   26.887
H   -6.471   42.241   25.561
H   -6.609   43.934   25.801
H   -7.719   44.012   22.108
H   -7.170   44.617   23.706
H   -7.299   42.905   23.439
H   -0.888   45.229   20.066
H   1.963   44.588   20.461
H   0.638   42.738   19.683
H   -0.635   43.269   18.826
H   -0.996   42.421   20.285
H   1.182   46.882   19.668
H   2.374   46.748   20.919
H   0.085   48.227   21.026
H   1.352   47.994   22.334
H   -0.058   47.042   22.338
H   5.548   45.414   25.225
H   5.215   44.807   23.693
H   5.936   43.731   24.830
H   3.053   43.887   29.453
H   4.173   39.834   30.584
H   3.227   40.397   31.987
H   4.876   40.889   31.752
H   -2.169   39.666   32.880
H   -3.444   40.461   31.959
H   -4.636   41.292   33.718
H   -2.517   39.009   36.398
H   -1.909   40.689   36.541
H   -1.590   39.668   35.041
H   -4.484   40.227   37.195
H   -5.515   40.702   35.988
H   -3.492   42.832   36.632
H   -4.444   42.277   37.947
H   -5.946   42.796   35.406
H   -4.879   44.122   35.936
H   -6.697   42.808   37.876
H   -5.113   45.071   38.054
H   -5.933   44.184   39.387
H   -6.735   45.478   38.615
H   -7.763   44.514   35.505
H   -8.140   45.328   36.954
H   -9.996   44.178   36.835
H   -9.006   43.110   37.813
H   -8.251   42.098   35.391
H   -9.887   42.527   34.875
H   -9.027   40.860   37.315
H   -9.497   38.734   35.648
H   -8.045   39.423   35.640
H   -9.266   40.114   34.412
H   -11.420   39.727   36.416
H   -11.440   41.306   35.836
H   -12.341   41.830   38.028
H   -10.647   41.630   38.529
H   -11.832   40.394   40.016
H   -10.865   39.394   38.944
H   -13.061   39.099   37.562
H   -15.086   39.450   39.687
H   -14.859   40.195   38.249
H   -13.890   40.815   39.562
H   -12.974   37.191   38.361
H   -13.838   37.605   39.826
H   -12.026   37.520   39.788

