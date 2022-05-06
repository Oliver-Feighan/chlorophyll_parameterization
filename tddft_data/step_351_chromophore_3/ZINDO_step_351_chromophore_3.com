%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_351_chromophore_3 ZINDO

0 1
Mg   1.668   7.778   25.792
C   2.146   9.833   28.554
C   2.554   5.063   28.059
C   1.597   5.661   23.243
C   1.579   10.458   23.753
N   2.098   7.545   28.081
C   2.158   8.522   28.971
C   2.262   8.022   30.383
C   2.614   6.490   30.217
C   2.446   6.327   28.694
C   4.016   6.018   30.715
C   0.950   8.309   31.183
C   1.071   8.579   32.677
C   2.292   8.001   33.450
O   3.277   8.609   33.788
O   2.147   6.682   33.733
N   1.886   5.647   25.693
C   2.289   4.730   26.711
C   2.277   3.380   26.111
C   2.013   3.547   24.774
C   1.745   4.976   24.523
C   2.495   2.093   26.936
C   1.956   2.520   23.651
O   1.393   2.778   22.603
C   2.516   1.058   23.763
N   1.356   8.053   23.805
C   1.378   7.015   22.939
C   1.257   7.570   21.448
C   1.136   9.053   21.694
C   1.378   9.271   23.130
C   2.372   7.164   20.378
C   -0.138   9.626   20.992
C   0.055   11.015   20.308
N   1.786   9.770   26.030
C   1.661   10.792   25.128
C   1.880   12.072   25.781
C   2.026   11.730   27.127
C   1.981   10.339   27.239
C   1.721   13.488   25.345
C   2.179   12.293   28.412
O   2.296   13.454   28.794
C   2.192   11.042   29.437
C   3.358   11.096   30.443
O   4.464   10.732   30.319
O   2.862   11.498   31.641
C   3.769   11.610   32.768
C   3.015   6.062   34.660
C   2.156   5.094   35.425
C   1.773   5.087   36.771
C   2.434   6.069   37.744
C   0.775   4.061   37.318
C   1.300   3.181   38.494
C   1.868   1.843   38.059
C   0.906   0.588   38.206
C   0.207   0.238   36.837
C   1.710   -0.604   38.828
C   0.733   -1.857   39.255
C   1.119   -2.350   40.659
C   1.666   -3.825   40.629
C   0.564   -4.854   40.470
C   2.406   -4.163   41.982
C   3.961   -4.441   41.687
C   4.911   -3.328   42.230
C   5.783   -2.619   41.175
C   7.192   -2.147   41.743
C   5.066   -1.436   40.658
H   2.728   4.207   28.716
H   1.651   5.053   22.338
H   1.400   11.343   23.139
H   3.061   8.496   30.952
H   1.933   5.874   30.804
H   4.575   5.523   29.920
H   3.853   5.223   31.442
H   4.638   6.734   31.252
H   0.367   7.393   31.088
H   0.368   9.068   30.661
H   0.240   8.156   33.240
H   0.876   9.612   32.963
H   3.444   1.627   26.671
H   1.620   1.461   26.782
H   2.614   2.212   28.013
H   1.692   0.716   24.390
H   3.452   0.982   24.316
H   2.620   0.464   22.855
H   0.342   7.097   21.092
H   2.004   9.552   21.264
H   2.527   8.056   19.771
H   1.994   6.312   19.814
H   3.232   6.966   21.019
H   -0.915   9.544   21.752
H   -0.528   9.008   20.182
H   -0.570   11.093   19.419
H   1.105   11.106   20.027
H   -0.229   11.863   20.932
H   0.679   13.606   25.049
H   2.470   13.591   24.561
H   2.020   14.198   26.117
H   1.281   11.173   30.021
H   4.422   12.463   32.580
H   4.566   10.894   32.967
H   3.231   11.860   33.682
H   3.483   6.772   35.342
H   3.748   5.409   34.187
H   1.710   4.395   34.717
H   3.362   5.534   37.946
H   1.813   6.327   38.603
H   2.589   7.021   37.237
H   0.453   3.391   36.521
H   -0.051   4.651   37.715
H   0.512   3.166   39.248
H   2.098   3.664   39.058
H   2.826   1.742   38.569
H   2.144   1.969   37.012
H   0.039   0.775   38.840
H   0.726   -0.511   36.238
H   0.107   1.047   36.113
H   -0.762   -0.212   37.051
H   2.206   -0.236   39.726
H   2.449   -1.047   38.160
H   0.912   -2.704   38.593
H   -0.304   -1.527   39.197
H   0.236   -2.221   41.284
H   1.833   -1.702   41.167
H   2.279   -3.923   39.733
H   -0.030   -5.003   41.372
H   0.993   -5.785   40.100
H   -0.139   -4.501   39.716
H   1.973   -5.008   42.517
H   2.321   -3.276   42.609
H   4.094   -4.318   40.612
H   4.221   -5.432   42.060
H   5.529   -3.810   42.986
H   4.295   -2.601   42.759
H   5.963   -3.266   40.317
H   8.084   -2.487   41.218
H   7.229   -2.454   42.788
H   7.265   -1.074   41.922
H   4.458   -1.760   39.814
H   5.785   -0.811   40.127
H   4.479   -0.810   41.330
