%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_901_chromophore_3 ZINDO

0 1
Mg   1.448   8.002   26.485
C   1.937   10.222   29.109
C   2.188   5.357   28.656
C   1.532   6.081   23.779
C   1.289   10.785   24.363
N   1.987   7.776   28.631
C   2.028   8.890   29.520
C   2.015   8.398   30.980
C   2.299   6.836   30.757
C   2.085   6.636   29.276
C   3.625   6.276   31.202
C   0.718   8.761   31.719
C   0.572   7.995   33.093
C   1.816   7.610   33.872
O   2.695   8.392   34.212
O   1.661   6.255   34.272
N   1.821   6.009   26.303
C   2.192   5.086   27.262
C   2.297   3.809   26.633
C   2.117   3.987   25.222
C   1.821   5.439   25.045
C   2.516   2.495   27.385
C   2.349   3.004   24.109
O   2.266   3.322   22.914
C   2.687   1.597   24.419
N   1.274   8.354   24.320
C   1.285   7.379   23.458
C   1.169   7.916   21.949
C   1.156   9.450   22.178
C   1.255   9.534   23.722
C   2.332   7.296   21.029
C   -0.107   10.119   21.623
C   0.117   11.133   20.476
N   1.532   10.135   26.610
C   1.413   11.089   25.677
C   1.570   12.414   26.253
C   1.782   12.127   27.666
C   1.761   10.732   27.828
C   1.522   13.799   25.563
C   1.932   12.744   28.933
O   2.033   13.923   29.219
C   1.916   11.493   29.920
C   3.103   11.668   30.711
O   4.280   11.614   30.222
O   2.814   11.610   32.084
C   3.904   11.834   33.070
C   2.665   5.875   35.308
C   1.995   4.864   36.218
C   1.993   4.926   37.564
C   2.590   6.186   38.306
C   0.982   4.049   38.354
C   0.951   2.559   37.914
C   0.602   1.485   39.024
C   1.643   0.363   39.031
C   2.940   0.966   39.604
C   1.067   -0.867   39.686
C   1.510   -2.258   39.096
C   2.189   -3.302   40.031
C   3.699   -3.427   39.940
C   4.202   -4.808   39.566
C   4.396   -2.856   41.207
C   5.841   -2.491   41.016
C   6.133   -0.963   41.221
C   7.570   -0.473   40.773
C   8.504   -0.830   41.970
C   7.511   1.038   40.409
H   2.321   4.523   29.349
H   1.600   5.500   22.857
H   1.165   11.670   23.735
H   2.816   8.901   31.522
H   1.495   6.321   31.283
H   4.238   5.897   30.384
H   3.510   5.458   31.912
H   4.173   7.143   31.572
H   -0.097   8.473   31.055
H   0.870   9.833   31.848
H   0.017   7.084   32.873
H   -0.058   8.567   33.775
H   2.662   2.618   28.458
H   3.370   1.949   26.984
H   1.671   1.807   27.405
H   2.590   0.902   23.585
H   2.098   1.186   25.239
H   3.752   1.621   24.649
H   0.246   7.528   21.518
H   2.020   9.870   21.662
H   2.979   8.156   20.859
H   2.093   6.792   20.093
H   2.942   6.610   21.618
H   -0.664   10.701   22.357
H   -0.779   9.319   21.313
H   -0.148   10.554   19.592
H   1.157   11.457   20.465
H   -0.509   12.011   20.634
H   0.703   13.930   24.857
H   2.495   13.886   25.078
H   1.557   14.760   26.076
H   1.014   11.522   30.531
H   4.907   11.994   32.674
H   3.983   10.963   33.721
H   3.653   12.673   33.719
H   2.795   6.777   35.907
H   3.591   5.689   34.764
H   1.398   4.066   35.776
H   1.833   6.967   38.245
H   3.503   6.594   37.872
H   2.773   6.108   39.377
H   0.016   4.459   38.059
H   1.107   4.031   39.436
H   1.788   2.243   37.290
H   0.009   2.442   37.378
H   -0.394   1.126   38.767
H   0.575   1.966   40.002
H   1.905   0.160   37.993
H   2.906   2.054   39.661
H   3.745   0.783   38.892
H   3.245   0.640   40.598
H   -0.022   -0.859   39.652
H   1.348   -0.855   40.739
H   2.189   -1.871   38.337
H   0.755   -2.729   38.465
H   1.655   -4.252   40.002
H   2.040   -3.008   41.070
H   3.974   -2.843   39.062
H   4.621   -4.821   38.559
H   3.389   -5.533   39.576
H   4.921   -5.107   40.328
H   4.353   -3.396   42.153
H   3.928   -1.936   41.556
H   6.391   -2.782   40.121
H   6.322   -2.882   41.913
H   5.934   -0.795   42.280
H   5.419   -0.416   40.605
H   7.961   -1.121   39.989
H   9.356   -1.446   41.684
H   7.972   -1.361   42.759
H   8.966   0.044   42.429
H   6.492   1.315   40.139
H   8.017   1.174   39.453
H   7.946   1.683   41.172

