%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_451_chromophore_23 TDDFT with cam-b3lyp functional

0 1
Mg   -10.651   40.819   42.423
C   -9.287   37.835   41.262
C   -8.439   42.549   40.467
C   -12.367   43.633   43.067
C   -13.250   38.896   43.908
N   -9.052   40.239   41.057
C   -8.663   38.978   40.798
C   -7.378   38.872   39.942
C   -7.058   40.452   39.809
C   -8.288   41.098   40.467
C   -5.717   40.859   40.431
C   -7.574   38.050   38.607
C   -6.761   36.830   38.342
C   -5.868   36.950   37.166
O   -4.675   37.024   37.186
O   -6.565   36.881   35.998
N   -10.479   42.907   41.875
C   -9.430   43.332   41.080
C   -9.561   44.765   41.027
C   -10.655   45.126   41.882
C   -11.222   43.900   42.340
C   -8.488   45.696   40.477
C   -11.109   46.582   42.222
O   -10.418   47.521   41.955
C   -12.417   46.807   43.010
N   -12.460   41.243   43.513
C   -12.969   42.450   43.578
C   -14.296   42.504   44.286
C   -14.712   40.967   44.253
C   -13.359   40.328   43.872
C   -14.346   43.156   45.690
C   -15.811   40.594   43.128
C   -17.183   40.134   43.698
N   -11.055   38.832   42.788
C   -12.153   38.169   43.395
C   -11.926   36.727   43.385
C   -10.919   36.568   42.427
C   -10.346   37.822   42.210
C   -12.826   35.662   43.893
C   -10.230   35.579   41.723
O   -10.400   34.340   41.667
C   -9.091   36.332   41.006
C   -7.807   35.724   41.640
O   -7.227   36.222   42.621
O   -7.501   34.599   40.920
C   -6.252   34.062   41.417
C   -5.865   36.858   34.713
C   -6.773   36.994   33.520
C   -7.726   37.941   33.426
C   -7.868   39.143   34.316
C   -8.718   37.811   32.314
C   -8.383   38.704   31.124
C   -8.072   37.807   29.893
C   -6.674   38.119   29.253
C   -6.130   36.890   28.422
C   -6.767   39.399   28.398
C   -5.863   40.546   28.922
C   -6.762   41.684   29.401
C   -5.914   42.827   30.075
C   -5.750   42.666   31.594
C   -6.488   44.299   29.697
C   -5.905   44.785   28.380
C   -5.143   46.088   28.426
C   -3.738   45.901   27.743
C   -2.599   46.620   28.625
C   -3.857   46.388   26.276
H   -7.717   43.036   39.808
H   -12.997   44.498   43.282
H   -14.054   38.355   44.412
H   -6.563   38.453   40.534
H   -6.940   40.656   38.745
H   -5.918   41.927   40.521
H   -4.950   40.704   39.672
H   -5.496   40.322   41.353
H   -7.385   38.720   37.768
H   -8.620   37.790   38.443
H   -7.509   36.049   38.203
H   -6.272   36.644   39.298
H   -7.484   45.273   40.518
H   -8.306   46.662   40.948
H   -8.638   45.955   39.429
H   -12.321   46.296   43.968
H   -13.316   46.511   42.469
H   -12.512   47.890   43.092
H   -15.095   43.035   43.768
H   -15.042   40.770   45.273
H   -13.311   43.409   45.921
H   -14.723   42.364   46.337
H   -14.858   44.118   45.707
H   -15.474   39.794   42.468
H   -16.071   41.495   42.572
H   -18.009   40.318   43.011
H   -17.394   40.802   44.533
H   -17.153   39.068   43.923
H   -13.768   35.598   43.349
H   -12.906   35.951   44.941
H   -12.325   34.694   43.889
H   -9.189   36.045   39.959
H   -6.565   33.439   42.255
H   -5.485   34.740   41.793
H   -5.824   33.451   40.622
H   -5.254   35.956   34.676
H   -5.245   37.745   34.589
H   -6.771   36.159   32.820
H   -8.758   39.011   34.932
H   -7.992   40.056   33.734
H   -6.976   39.220   34.938
H   -9.691   38.119   32.697
H   -8.818   36.769   32.009
H   -7.578   39.415   31.307
H   -9.252   39.340   30.954
H   -8.885   37.961   29.183
H   -8.053   36.742   30.125
H   -6.115   38.327   30.165
H   -5.831   37.070   27.390
H   -6.901   36.124   28.343
H   -5.356   36.302   28.915
H   -7.786   39.732   28.201
H   -6.548   39.084   27.377
H   -5.302   40.785   28.018
H   -5.119   40.327   29.687
H   -7.600   41.440   30.054
H   -7.368   42.079   28.586
H   -4.919   42.768   29.634
H   -6.248   41.812   32.054
H   -6.024   43.546   32.176
H   -4.671   42.652   31.749
H   -6.381   45.023   30.505
H   -7.532   44.150   29.421
H   -6.705   45.034   27.682
H   -5.224   44.034   27.980
H   -5.048   46.430   29.457
H   -5.633   46.955   27.983
H   -3.419   44.864   27.845
H   -1.719   46.740   27.994
H   -2.280   46.090   29.523
H   -2.948   47.556   29.060
H   -3.116   47.183   26.190
H   -4.753   46.986   26.109
H   -3.675   45.537   25.620
